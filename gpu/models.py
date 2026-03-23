"""
Arche Models — PyTorch (GPU version)
======================================
Both Autoregressive and MDLM models for H100 training.
Scaled up: 300M params AR, ~200M params MDLM.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared Components
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, offset=0):
        # x: (B, n_heads, T, head_dim)
        T = x.shape[2]
        t = torch.arange(offset, offset + T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        cos_emb = emb.cos()[None, None, :, :]  # (1, 1, T, dim)
        sin_emb = emb.sin()[None, None, :, :]

        # Apply rotary embedding
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos_emb + rotated * sin_emb


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =============================================================================
# Autoregressive Model (Causal Transformer)
# =============================================================================

class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, rope):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner = n_heads * head_dim

        self.wq = nn.Linear(d_model, inner, bias=False)
        self.wk = nn.Linear(d_model, inner, bias=False)
        self.wv = nn.Linear(d_model, inner, bias=False)
        self.wo = nn.Linear(inner, d_model, bias=False)
        self.rope = rope

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        # Use PyTorch's efficient SDPA with causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class ARBlock(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, ffn_dim, rope):
        super().__init__()
        self.attn = CausalAttention(d_model, n_heads, head_dim, rope)
        self.ffn = SwiGLUFFN(d_model, ffn_dim)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AutoregressiveModel(nn.Module):
    """Standard causal transformer for language modeling."""

    def __init__(self, vocab_size, d_model=768, n_heads=12, head_dim=64,
                 ffn_dim=2048, n_layers=12):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        rope = RotaryEmbedding(head_dim)

        self.blocks = nn.ModuleList([
            ARBlock(d_model, n_heads, head_dim, ffn_dim, rope)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        # Weight tying
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens):
        """tokens: (B, T) → logits: (B, T, vocab_size)"""
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

    def compute_loss(self, tokens):
        """Standard next-token prediction loss."""
        logits = self(tokens[:, :-1])
        targets = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        return loss


# =============================================================================
# MDLM (Masked Diffusion Language Model)
# =============================================================================

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Linear(d_model, d_model)

    def forward(self, t):
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return F.silu(self.mlp(emb))


class BidirectionalAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, rope):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim

        self.wq = nn.Linear(d_model, inner, bias=False)
        self.wk = nn.Linear(d_model, inner, bias=False)
        self.wv = nn.Linear(d_model, inner, bias=False)
        self.wo = nn.Linear(inner, d_model, bias=False)
        self.rope = rope

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        # NO causal mask — bidirectional
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class DiffusionBlock(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, ffn_dim, rope):
        super().__init__()
        self.attn = BidirectionalAttention(d_model, n_heads, head_dim, rope)
        self.ffn = SwiGLUFFN(d_model, ffn_dim)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MDLMModel(nn.Module):
    """Masked Diffusion Language Model."""

    def __init__(self, vocab_size, d_model=768, n_heads=12, head_dim=64,
                 ffn_dim=2048, n_layers=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size  # Extra token for [MASK]
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size + 1, d_model)  # +1 for MASK
        self.timestep_embed = TimestepEmbedding(d_model)
        rope = RotaryEmbedding(head_dim)

        self.blocks = nn.ModuleList([
            DiffusionBlock(d_model, n_heads, head_dim, ffn_dim, rope)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, t):
        """tokens: (B, T) with possible MASK ids, t: (B,) → logits: (B, T, V)"""
        x = self.embed(tokens) + self.timestep_embed(t)[:, None, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def compute_loss(self, clean_tokens):
        """Mask at random timestep, predict originals."""
        B, T = clean_tokens.shape
        device = clean_tokens.device

        # Random timestep per example
        t = torch.rand(B, device=device)

        # Cosine mask rate
        mask_rate = 1.0 - (torch.cos(t * math.pi / 2) ** 2)

        # Create mask
        rand = torch.rand(B, T, device=device)
        mask = rand < mask_rate[:, None]

        # Apply mask
        noised = clean_tokens.clone()
        noised[mask] = self.mask_token_id

        # Predict
        logits = self(noised, t)

        # Loss on masked positions only
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = clean_tokens.reshape(-1)
        mask_flat = mask.reshape(-1)

        per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = (per_token * mask_flat.float()).sum() / mask_flat.float().sum().clamp(min=1)
        return loss

    @torch.no_grad()
    def generate(self, length=256, steps=100, temperature=0.8, rep_penalty=1.3,
                 device='cuda'):
        """Generate by iterative demasking."""
        x = torch.full((1, length), self.mask_token_id, dtype=torch.long, device=device)
        is_masked = torch.ones(length, dtype=torch.bool, device=device)
        token_counts = {}

        for step in range(steps):
            n_masked = is_masked.sum().item()
            if n_masked == 0:
                break

            t_val = 1.0 - step / steps
            temp = temperature * (0.4 + 0.6 * t_val)  # Anneal
            t = torch.tensor([t_val], device=device)

            logits = self(x, t)[0]  # (length, V)

            # Repetition penalty
            if rep_penalty > 1.0 and token_counts:
                for tok_id, count in token_counts.items():
                    logits[:, tok_id] /= min(rep_penalty ** count, 5.0)

            logits /= max(temp, 0.1)
            probs = F.softmax(logits, dim=-1)

            sampled = torch.multinomial(probs, 1).squeeze(-1)  # (length,)
            confidence = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
            confidence[~is_masked] = 0.0

            # Unmask most confident
            steps_left = steps - step
            n_unmask = max(1, n_masked // max(steps_left, 1))

            _, top_idx = confidence.topk(min(n_unmask, n_masked))
            for idx in top_idx:
                i = idx.item()
                tok = sampled[i].item()
                x[0, i] = tok
                is_masked[i] = False
                token_counts[tok] = token_counts.get(tok, 0) + 1

        # Fill remaining
        if is_masked.any():
            logits = self(x, torch.tensor([0.0], device=device))[0]
            x[0, is_masked] = logits[is_masked].argmax(dim=-1)

        return x[0]


# =============================================================================
# Model Factory
# =============================================================================

def create_models(vocab_size, scale="300M"):
    """Create both AR and MDLM models at the specified scale."""
    configs = {
        "100M": dict(d_model=512, n_heads=8, head_dim=64, ffn_dim=1024, n_layers=9),
        "300M": dict(d_model=768, n_heads=12, head_dim=64, ffn_dim=2048, n_layers=12),
        "700M": dict(d_model=1024, n_heads=16, head_dim=64, ffn_dim=2816, n_layers=16),
    }

    cfg = configs.get(scale, configs["300M"])
    ar = AutoregressiveModel(vocab_size, **cfg)
    mdlm = MDLMModel(vocab_size, **cfg)

    ar_params = sum(p.numel() for p in ar.parameters())
    mdlm_params = sum(p.numel() for p in mdlm.parameters())
    print(f"Scale: {scale}")
    print(f"  AR:   {ar_params/1e6:.1f}M params")
    print(f"  MDLM: {mdlm_params/1e6:.1f}M params")

    return ar, mdlm, cfg
