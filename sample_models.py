"""Generate samples from trained AR and MDLM checkpoints (H100)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_SAMPLES = 10
LENGTH = 128

# ── Model definitions matching checkpoint key names ─────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x):
        T = x.shape[2]
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_e = emb.cos()[None, None, :, :]
        sin_e = emb.sin()[None, None, :, :]
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return x * cos_e + torch.cat([-x2, x1], dim=-1) * sin_e

class SwiGLUFFN(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)
        self.w3 = nn.Linear(d, ff, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class ARBlock(nn.Module):
    def __init__(self, d, nh, hd, ff):
        super().__init__()
        inner = nh * hd
        self.wq = nn.Linear(d, inner, bias=False)
        self.wk = nn.Linear(d, inner, bias=False)
        self.wv = nn.Linear(d, inner, bias=False)
        self.wo = nn.Linear(inner, d, bias=False)
        self.rope = RotaryEmbedding(hd)
        self.ffn = SwiGLUFFN(d, ff)
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.nh, self.hd = nh, hd
    def forward(self, x):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B,T,self.nh,self.hd).transpose(1,2)
        k = self.wk(h).view(B,T,self.nh,self.hd).transpose(1,2)
        v = self.wv(h).view(B,T,self.nh,self.hd).transpose(1,2)
        q, k = self.rope(q), self.rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(o.transpose(1,2).reshape(B,T,-1))
        x = x + self.ffn(self.n2(x))
        return x

class ARModel(nn.Module):
    def __init__(self, V, d=768, nh=12, hd=64, ff=2048, nl=12):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([ARBlock(d, nh, hd, ff) for _ in range(nl)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
    def forward(self, tokens):
        x = self.emb(tokens)
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))

class MDLMBlock(nn.Module):
    def __init__(self, d, nh, hd, ff):
        super().__init__()
        inner = nh * hd
        self.wq = nn.Linear(d, inner, bias=False)
        self.wk = nn.Linear(d, inner, bias=False)
        self.wv = nn.Linear(d, inner, bias=False)
        self.wo = nn.Linear(inner, d, bias=False)
        self.rope = RotaryEmbedding(hd)
        self.ffn = SwiGLUFFN(d, ff)
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.nh, self.hd = nh, hd
    def forward(self, x):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B,T,self.nh,self.hd).transpose(1,2)
        k = self.wk(h).view(B,T,self.nh,self.hd).transpose(1,2)
        v = self.wv(h).view(B,T,self.nh,self.hd).transpose(1,2)
        q, k = self.rope(q), self.rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # bidirectional
        x = x + self.wo(o.transpose(1,2).reshape(B,T,-1))
        x = x + self.ffn(self.n2(x))
        return x

class TimestepEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Linear(d, d)
    def forward(self, t):
        half = self.d // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return F.silu(self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1)))

class MDLMModel(nn.Module):
    def __init__(self, V, d=768, nh=12, hd=64, ff=2048, nl=12):
        super().__init__()
        self.V = V
        self.mask_id = V  # extra token
        self.emb = nn.Embedding(V + 1, d)
        self.t_emb = TimestepEmb(d)
        self.blocks = nn.ModuleList([MDLMBlock(d, nh, hd, ff) for _ in range(nl)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
    def forward(self, tokens, t):
        x = self.emb(tokens) + self.t_emb(t)[:, None, :]
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))
    @torch.no_grad()
    def generate(self, length=256, steps=100, temperature=0.8, rep_penalty=1.3, device='cpu'):
        x = torch.full((1, length), self.mask_id, dtype=torch.long, device=device)
        is_masked = torch.ones(length, dtype=torch.bool, device=device)
        token_counts = {}
        for step in range(steps):
            n_masked = is_masked.sum().item()
            if n_masked == 0: break
            t_val = 1.0 - step / steps
            temp = temperature * (0.4 + 0.6 * t_val)
            t = torch.tensor([t_val], device=device)
            logits = self(x, t)[0]
            if rep_penalty > 1.0 and token_counts:
                for tok_id, count in token_counts.items():
                    logits[:, tok_id] /= min(rep_penalty ** count, 5.0)
            logits /= max(temp, 0.1)
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            confidence = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
            confidence[~is_masked] = 0.0
            steps_left = steps - step
            n_unmask = max(1, n_masked // max(steps_left, 1))
            _, top_idx = confidence.topk(min(n_unmask, n_masked))
            for idx in top_idx:
                i = idx.item()
                tok = sampled[i].item()
                x[0, i] = tok
                is_masked[i] = False
                token_counts[tok] = token_counts.get(tok, 0) + 1
        if is_masked.any():
            logits = self(x, torch.tensor([0.0], device=device))[0]
            x[0, is_masked] = logits[is_masked].argmax(dim=-1)
        return x[0]

# ── Main ────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
V = 50257

# Load AR
print("\nLoading AR...")
ar = ARModel(V)
ar.load_state_dict(torch.load("/Users/caiovicentino/Downloads/AR_best.pt",
                               map_location=DEVICE, weights_only=True))
ar = ar.to(DEVICE).eval()
print(f"  OK ({sum(p.numel() for p in ar.parameters())/1e6:.1f}M params)")

# Load MDLM
print("Loading MDLM...")
mdlm = MDLMModel(V)
mdlm.load_state_dict(torch.load("/Users/caiovicentino/Downloads/MDLM_best.pt",
                                  map_location=DEVICE, weights_only=True))
mdlm = mdlm.to(DEVICE).eval()
print(f"  OK ({sum(p.numel() for p in mdlm.parameters())/1e6:.1f}M params)")

# Generate AR
print(f"\n{'='*70}")
print(f"  AR SAMPLES")
print(f"{'='*70}")
with torch.no_grad():
    for i in range(N_SAMPLES):
        x = torch.tensor([[tokenizer.encode("Once")[0]]], device=DEVICE)
        for _ in range(LENGTH - 1):
            logits = ar(x)
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            x = torch.cat([x, next_tok], dim=1)
        text = tokenizer.decode(x[0].tolist())
        print(f"\n[AR-{i+1}] {text[:300]}")

# Generate MDLM
print(f"\n{'='*70}")
print(f"  MDLM SAMPLES")
print(f"{'='*70}")
with torch.no_grad():
    for i in range(N_SAMPLES):
        tokens = mdlm.generate(length=LENGTH, steps=100, temperature=0.8,
                                rep_penalty=1.3, device=DEVICE)
        text = tokenizer.decode(tokens.tolist())
        print(f"\n[MDLM-{i+1}] {text[:300]}")

print(f"\n{'='*70}")
print("Done!")
