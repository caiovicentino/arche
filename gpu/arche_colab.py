"""
=============================================================
  ARCHE — Full Paper Experiment (Google Colab Version)
=============================================================

Instructions:
  1. Open Google Colab (colab.research.google.com)
  2. Select GPU runtime: Runtime → Change runtime type → GPU (T4/A100)
  3. Upload this file OR copy-paste into cells
  4. Run!

For Colab Free (T4 16GB):  --scale 100M --steps 10000  (~1h)
For Colab Pro (A100 40GB): --scale 300M --steps 30000  (~2-3h)
=============================================================
"""

# ── Cell 1: Setup ────────────────────────────────────────────
# !pip install -q torch transformers datasets tqdm

import os, time, json, math
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Cell 2: Config ───────────────────────────────────────────

# CHANGE THESE based on your Colab tier:
SCALE = "100M"       # "100M" for free T4, "300M" for Pro A100
STEPS = 15000        # 10K-30K depending on time budget
BATCH_SIZE = 16      # 16 for T4, 32 for A100
SEQ_LEN = 256        # 256 for T4, 512 for A100
MAX_TOKENS = 20_000_000  # 20M for T4, 50M for A100
LR = 3e-4
WARMUP = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram:.1f} GB")
    # Auto-scale for H100/A100
    if vram >= 70:  # H100 80GB
        SCALE = "300M"
        BATCH_SIZE = 32
        SEQ_LEN = 512
        MAX_TOKENS = 50_000_000
        STEPS = 20000
        print(f"H100 detected! Scaling up: {SCALE}, batch={BATCH_SIZE}, seq={SEQ_LEN}")
    elif vram >= 35:  # A100 40GB
        SCALE = "300M"
        BATCH_SIZE = 16
        SEQ_LEN = 512
        MAX_TOKENS = 50_000_000


# ── Cell 3: Model Definitions ───────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv)
    def forward(self, x, offset=0):
        T = x.shape[2]
        t = torch.arange(offset, offset + T, device=x.device, dtype=self.inv_freq.dtype)
        f = torch.outer(t, self.inv_freq)
        e = torch.cat([f, f], dim=-1)
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return x * e.cos()[None,None] + torch.cat([-x2, x1], dim=-1) * e.sin()[None,None]

class SwiGLU(nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.w1 = nn.Linear(d, ffn, bias=False)
        self.w2 = nn.Linear(ffn, d, bias=False)
        self.w3 = nn.Linear(d, ffn, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TimestepEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Linear(d, d)
    def forward(self, t):
        h = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / h)
        a = t[:, None] * f[None, :]
        return F.silu(self.mlp(torch.cat([a.sin(), a.cos()], -1)))

# ── Autoregressive ──

class ARBlock(nn.Module):
    def __init__(self, d, nh, hd, ffn, rope):
        super().__init__()
        inner = nh * hd
        self.wq = nn.Linear(d, inner, bias=False)
        self.wk = nn.Linear(d, inner, bias=False)
        self.wv = nn.Linear(d, inner, bias=False)
        self.wo = nn.Linear(inner, d, bias=False)
        self.ffn = SwiGLU(d, ffn)
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.rope = rope
        self.nh, self.hd = nh, hd

    def forward(self, x):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.rope(self.wq(h).view(B,T,self.nh,self.hd).transpose(1,2))
        k = self.rope(self.wk(h).view(B,T,self.nh,self.hd).transpose(1,2))
        v = self.wv(h).view(B,T,self.nh,self.hd).transpose(1,2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(a.transpose(1,2).reshape(B,T,-1))
        x = x + self.ffn(self.n2(x))
        return x

class ARModel(nn.Module):
    def __init__(self, V, d=512, nh=8, hd=64, ffn=1024, nl=9):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        rope = RotaryEmbedding(hd)
        self.blocks = nn.ModuleList([ARBlock(d, nh, hd, ffn, rope) for _ in range(nl)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight  # Weight tying

    def forward(self, tok):
        x = self.emb(tok)
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))

    def loss(self, tok):
        return F.cross_entropy(self(tok[:,:-1]).reshape(-1, self(tok[:,:-1]).size(-1)),
                               tok[:,1:].reshape(-1))

    def loss(self, tok):
        logits = self(tok[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tok[:, 1:].reshape(-1))

# ── MDLM ──

class MDLMBlock(nn.Module):
    def __init__(self, d, nh, hd, ffn, rope):
        super().__init__()
        inner = nh * hd
        self.wq = nn.Linear(d, inner, bias=False)
        self.wk = nn.Linear(d, inner, bias=False)
        self.wv = nn.Linear(d, inner, bias=False)
        self.wo = nn.Linear(inner, d, bias=False)
        self.ffn = SwiGLU(d, ffn)
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.rope = rope
        self.nh, self.hd = nh, hd

    def forward(self, x):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.rope(self.wq(h).view(B,T,self.nh,self.hd).transpose(1,2))
        k = self.rope(self.wk(h).view(B,T,self.nh,self.hd).transpose(1,2))
        v = self.wv(h).view(B,T,self.nh,self.hd).transpose(1,2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # BIDIRECTIONAL
        x = x + self.wo(a.transpose(1,2).reshape(B,T,-1))
        x = x + self.ffn(self.n2(x))
        return x

class MDLMModel(nn.Module):
    def __init__(self, V, d=512, nh=8, hd=64, ffn=1024, nl=9):
        super().__init__()
        self.V = V
        self.mask_id = V
        self.emb = nn.Embedding(V + 1, d)
        self.t_emb = TimestepEmb(d)
        rope = RotaryEmbedding(hd)
        self.blocks = nn.ModuleList([MDLMBlock(d, nh, hd, ffn, rope) for _ in range(nl)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, tok, t):
        x = self.emb(tok) + self.t_emb(t)[:, None, :]
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))

    def loss(self, tok):
        B, T = tok.shape
        t = torch.rand(B, device=tok.device)
        rate = 1.0 - (torch.cos(t * math.pi / 2) ** 2)
        mask = torch.rand(B, T, device=tok.device) < rate[:, None]
        noised = tok.clone()
        noised[mask] = self.mask_id
        logits = self(noised, t).reshape(-1, self.V)
        targets = tok.reshape(-1)
        per_tok = F.cross_entropy(logits, targets, reduction='none')
        return (per_tok * mask.reshape(-1).float()).sum() / mask.float().sum().clamp(min=1)

    @torch.no_grad()
    def generate(self, length=256, steps=100, temp=0.8, rep_pen=1.3, device='cuda'):
        x = torch.full((1, length), self.mask_id, dtype=torch.long, device=device)
        is_masked = torch.ones(length, dtype=torch.bool, device=device)
        counts = {}
        for step in range(steps):
            nm = is_masked.sum().item()
            if nm == 0: break
            tv = 1.0 - step / steps
            t = torch.tensor([tv], device=device)
            logits = self(x, t)[0]
            if counts:
                for tid, c in counts.items():
                    logits[:, tid] /= min(rep_pen ** c, 5.0)
            logits /= max(temp * (0.4 + 0.6 * tv), 0.1)
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
            conf[~is_masked] = 0.0
            n_un = max(1, nm // max(steps - step, 1))
            _, top = conf.topk(min(n_un, nm))
            for i in top:
                tok = sampled[i.item()].item()
                x[0, i] = tok
                is_masked[i] = False
                counts[tok] = counts.get(tok, 0) + 1
        if is_masked.any():
            logits = self(x, torch.tensor([0.0], device=device))[0]
            x[0, is_masked] = logits[is_masked].argmax(dim=-1)
        return x[0]


# ── Cell 4: Data ─────────────────────────────────────────────

def load_data(max_tokens=MAX_TOKENS, seq_len=SEQ_LEN):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    all_tokens = []
    for ex in ds:
        story = ex["text"].strip()
        if story and len(story) > 50:
            all_tokens.extend(tokenizer.encode(story))
            if len(all_tokens) >= max_tokens:
                break
    print(f"Tokens: {len(all_tokens):,}")

    tokens = np.array(all_tokens[:max_tokens], dtype=np.int64)
    n = len(tokens) // seq_len
    chunks = tokens[:n * seq_len].reshape(n, seq_len)

    n_val = max(50, n // 20)
    perm = np.random.RandomState(42).permutation(n)
    train = torch.tensor(chunks[perm[:-n_val]])
    val = torch.tensor(chunks[perm[-n_val:]])
    print(f"Train: {len(train):,} | Val: {len(val):,} | Seq: {seq_len}")
    return train, val, tokenizer


# ── Cell 5: Training ─────────────────────────────────────────

def train_one(model, train_data, val_data, name, steps, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True)

    print(f"\n{'='*50}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Training {name} ({n_params/1e6:.1f}M params)")
    print(f"  Steps: {steps} | Batch: {BATCH_SIZE} | Device: {device}")
    print(f"{'='*50}\n")

    best_val = float("inf")
    start = time.time()
    step = 0
    history = []

    while step < steps:
        for (batch,) in loader:
            if step >= steps: break
            step += 1

            # LR schedule
            if step < WARMUP:
                lr = LR * step / WARMUP
            else:
                p = (step - WARMUP) / max(steps - WARMUP, 1)
                lr = LR * 0.05 + 0.95 * LR * 0.5 * (1 + math.cos(math.pi * p))
            for pg in opt.param_groups: pg["lr"] = lr

            batch = batch.to(device)
            loss = model.loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 100 == 0:
                elapsed = time.time() - start
                tps = step * BATCH_SIZE * SEQ_LEN / elapsed
                print(f"  [{name}] {step:6d}/{steps} | loss {loss.item():.4f} | "
                      f"lr {lr:.2e} | {tps:,.0f} tok/s")

            if step % 2000 == 0 or step == steps:
                model.eval()
                vl = 0; vc = 0
                with torch.no_grad():
                    for i in range(0, min(len(val_data), 50*BATCH_SIZE), BATCH_SIZE):
                        b = val_data[i:i+BATCH_SIZE].to(device)
                        if b.shape[0] < 2: continue
                        vl += model.loss(b).item(); vc += 1
                vl /= max(vc, 1)
                model.train()
                is_best = vl < best_val
                if is_best:
                    best_val = vl
                    torch.save(model.state_dict(), f"{name}_best.pt")
                history.append((step, vl))
                print(f"  [{name}] >> val={vl:.4f} {'★ BEST' if is_best else ''}")

                # Sample from MDLM
                if hasattr(model, 'generate'):
                    model.eval()
                    tok = model.generate(length=128, steps=80, device=device)
                    from transformers import AutoTokenizer
                    t = AutoTokenizer.from_pretrained("gpt2")
                    print(f"  [{name}] >> \"{t.decode(tok.tolist())[:200]}\"")
                    model.train()
                print()

    elapsed = time.time() - start
    tps = steps * BATCH_SIZE * SEQ_LEN / elapsed
    print(f"  [{name}] Done: best_val={best_val:.4f} | {elapsed/60:.1f}min | {tps:,.0f} tok/s")
    return {"name": name, "best_val": best_val, "time_min": elapsed/60,
            "tok_s": tps, "history": history,
            "params": sum(p.numel() for p in model.parameters())}


# ── Cell 6: Run Everything ───────────────────────────────────

def main():
    CONFIGS = {
        "100M": dict(d=512, nh=8, hd=64, ffn=1024, nl=9),
        "300M": dict(d=768, nh=12, hd=64, ffn=2048, nl=12),
    }
    cfg = CONFIGS[SCALE]
    print(f"Scale: {SCALE} | Config: {cfg}")

    # Data
    train_data, val_data, tokenizer = load_data()
    V = tokenizer.vocab_size

    # Create models
    ar = ARModel(V, d=cfg["d"], nh=cfg["nh"], hd=cfg["hd"], ffn=cfg["ffn"], nl=cfg["nl"])
    mdlm = MDLMModel(V, d=cfg["d"], nh=cfg["nh"], hd=cfg["hd"], ffn=cfg["ffn"], nl=cfg["nl"])

    # Train both
    ar_res = train_one(ar, train_data, val_data, "AR", STEPS, DEVICE)
    mdlm_res = train_one(mdlm, train_data, val_data, "MDLM", STEPS, DEVICE)

    # Final comparison
    print(f"\n{'='*60}")
    print(f"  RESULTS — {SCALE} scale, {STEPS} steps")
    print(f"{'='*60}")
    print(f"  {'':25s} {'AR':>15} {'MDLM':>15}")
    print(f"  {'-'*23:25s} {'-'*13:>15} {'-'*13:>15}")
    print(f"  {'Parameters':25s} {ar_res['params']/1e6:>14.1f}M {mdlm_res['params']/1e6:>14.1f}M")
    print(f"  {'Best val loss':25s} {ar_res['best_val']:>15.4f} {mdlm_res['best_val']:>15.4f}")
    print(f"  {'Training time':25s} {ar_res['time_min']:>13.1f}m {mdlm_res['time_min']:>13.1f}m")
    print(f"  {'Throughput':25s} {ar_res['tok_s']:>12,.0f} {mdlm_res['tok_s']:>12,.0f}")

    # Generate final samples
    print(f"\n--- AR Samples ---")
    ar.load_state_dict(torch.load("AR_best.pt", weights_only=True))
    ar = ar.to(DEVICE).eval()
    for i in range(3):
        x = torch.tensor([[tokenizer.encode("Once upon a time")[0]]], device=DEVICE)
        with torch.no_grad():
            for _ in range(200):
                logits = ar(x)[:, -1, :] / 0.8
                x = torch.cat([x, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        print(f"  [{i+1}] {tokenizer.decode(x[0].tolist())[:250]}\n")

    print(f"\n--- MDLM Samples ---")
    mdlm.load_state_dict(torch.load("MDLM_best.pt", weights_only=True))
    mdlm = mdlm.to(DEVICE).eval()
    for i in range(3):
        tok = mdlm.generate(length=200, steps=100, device=DEVICE)
        print(f"  [{i+1}] {tokenizer.decode(tok.tolist())[:250]}\n")

    # Save
    results = {"scale": SCALE, "steps": STEPS, "ar": ar_res, "mdlm": mdlm_res}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
