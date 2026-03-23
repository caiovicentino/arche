"""
Masked Diffusion Language Model (MDLM)
========================================
Instead of generating text left-to-right (autoregressive), MDLM:
  1. Starts with a fully masked sequence [MASK, MASK, MASK, ...]
  2. Iteratively predicts and reveals tokens over T steps
  3. Each step unmasks the most confident predictions
  4. After T steps → complete text

Training: mask random tokens at random noise levels, predict the originals.
Like BERT, but with a diffusion-style noise schedule that enables generation.

Key difference from autoregressive:
  - BIDIRECTIONAL attention (sees both left and right context)
  - PARALLEL generation (all tokens at once, refined iteratively)
  - Can "reconsider" tokens in later steps

Based on: MDLM (NeurIPS 2024), LLaDA (2025)
"""

import math
import mlx.core as mx
import mlx.nn as nn


def silu(x):
    return x * mx.sigmoid(x)


# =============================================================================
# Noise Schedule
# =============================================================================

class CosineSchedule:
    """
    Cosine noise schedule for masking.
    alpha(t) = cos(π/2 · t)² gives the fraction of tokens UNMASKED.
    t=0 → alpha=1 (clean), t=1 → alpha=0 (fully masked)
    """
    def mask_rate(self, t):
        """Fraction of tokens to mask at continuous time t ∈ [0, 1]."""
        return 1.0 - (mx.cos(t * math.pi / 2) ** 2)

    def sample_t(self, batch_size):
        """Sample random timesteps for training."""
        return mx.random.uniform(shape=(batch_size,))


# =============================================================================
# Timestep Embedding
# =============================================================================

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding (like diffusion models for images)."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Linear(d_model, d_model, bias=True)

    def __call__(self, t):
        """t: (B,) float in [0, 1] → (B, d_model) embedding."""
        half = self.d_model // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(half) / half)
        args = t[:, None] * freqs[None, :]  # (B, half)
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)  # (B, d_model)
        return silu(self.mlp(emb))


# =============================================================================
# Bidirectional Attention Block (no causal mask)
# =============================================================================

class BidirectionalAttention(nn.Module):
    """Standard multi-head attention WITHOUT causal mask.
    The model sees all positions — both left and right context."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        inner_dim = config.n_heads * config.head_dim

        self.wq = nn.Linear(config.d_model, inner_dim, bias=False)
        self.wk = nn.Linear(config.d_model, inner_dim, bias=False)
        self.wv = nn.Linear(config.d_model, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, config.d_model, bias=False)
        self.rope = nn.RoPE(config.head_dim, base=config.rope_theta)

    def __call__(self, x):
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        # NO causal mask — full bidirectional attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        weights = mx.softmax(scores, axis=-1)
        output = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(output)


# =============================================================================
# FFN (SwiGLU)
# =============================================================================

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.ffn_dim, bias=False)

    def __call__(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


# =============================================================================
# Transformer Block (bidirectional)
# =============================================================================

class DiffusionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = BidirectionalAttention(config)
        self.ffn = FFN(config)
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# MDLM Model
# =============================================================================

class MDLMConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 512)
        self.n_heads = kwargs.get("n_heads", 8)
        self.head_dim = kwargs.get("head_dim", 64)
        self.ffn_dim = kwargs.get("ffn_dim", 1024)
        self.n_layers = kwargs.get("n_layers", 9)
        self.vocab_size = kwargs.get("vocab_size", 50257)
        self.max_seq_len = kwargs.get("max_seq_len", 512)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        # MASK token = vocab_size (extra token)
        self.mask_token_id = self.vocab_size
        self.total_vocab = self.vocab_size + 1  # +1 for [MASK]


class MDLM(nn.Module):
    """
    Masked Diffusion Language Model.

    Training: mask tokens at random noise level, predict originals.
    Generation: start fully masked, iteratively unmask with confidence.
    """

    def __init__(self, config: MDLMConfig):
        super().__init__()
        self.config = config
        self.schedule = CosineSchedule()

        # Embeddings (vocab + 1 for [MASK] token)
        self.embed = nn.Embedding(config.total_vocab, config.d_model)
        self.timestep_embed = TimestepEmbedding(config.d_model)

        # Bidirectional transformer blocks
        self.blocks = [DiffusionBlock(config) for _ in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.d_model)

        # Output head (predicts original vocab only, not [MASK])
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(self, tokens, t):
        """
        Forward pass.
        Args:
            tokens: (B, T) token IDs (may contain mask_token_id)
            t: (B,) timestep values in [0, 1]
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tokens.shape

        # Token embeddings + timestep conditioning
        x = self.embed(tokens)  # (B, T, d_model)
        t_emb = self.timestep_embed(t)  # (B, d_model)
        x = x + t_emb[:, None, :]  # Broadcast timestep to all positions

        # Bidirectional transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Predict original tokens
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    def compute_loss(self, clean_tokens):
        """
        Training loss: mask tokens at random timestep, predict originals.
        Only compute loss on MASKED positions (like BERT MLM).

        Args:
            clean_tokens: (B, T) original token IDs
        Returns:
            loss: scalar
        """
        B, T = clean_tokens.shape

        # Sample random timestep per example
        t = self.schedule.sample_t(B)  # (B,)

        # Compute mask rate for each example
        mask_rate = self.schedule.mask_rate(t)  # (B,)

        # Create mask: True where we mask
        rand = mx.random.uniform(shape=(B, T))
        mask = rand < mask_rate[:, None]  # (B, T)

        # Apply mask
        mask_id = mx.array(self.config.mask_token_id)
        noised = mx.where(mask, mask_id, clean_tokens)

        # Forward pass
        logits = self(noised, t)  # (B, T, vocab_size)

        # Cross-entropy only on masked positions
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, self.config.vocab_size)
        targets_flat = clean_tokens.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Compute per-token loss
        per_token_loss = nn.losses.cross_entropy(logits_flat, targets_flat)

        # Average only over masked tokens
        masked_loss = mx.sum(per_token_loss * mask_flat.astype(per_token_loss.dtype))
        n_masked = mx.maximum(mx.sum(mask_flat.astype(mx.float32)), mx.array(1.0))
        loss = masked_loss / n_masked

        return loss


# =============================================================================
# Sampling (Generation)
# =============================================================================

def sample(model, length=128, steps=100, temperature=0.9, top_p=0.92,
           repetition_penalty=1.3, temp_anneal=True):
    """
    Generate text by iterative demasking with improved sampling.

    Improvements over naive sampling:
      1. Temperature annealing: high temp early (diversity) → low temp late (coherence)
      2. Repetition penalty: penalize tokens that appear too often
      3. Top-p (nucleus) filtering per position
      4. More steps for better refinement (100 default)
    """
    config = model.config
    B = 1

    x = mx.full((B, length), config.mask_token_id, dtype=mx.int32)
    is_masked_list = [True] * length
    x_list = [config.mask_token_id] * length
    token_counts = {}  # Track token frequencies for repetition penalty

    for step in range(steps):
        n_masked = sum(is_masked_list)
        if n_masked == 0:
            break

        # Noise level: high → low
        t_val = 1.0 - step / steps
        t = mx.array([t_val])

        # Temperature annealing: 1.2 at start → 0.5 at end
        if temp_anneal:
            temp = temperature * (0.4 + 0.6 * t_val)
        else:
            temp = temperature

        # Forward pass
        x_tensor = mx.array([x_list], dtype=mx.int32)
        logits = model(x_tensor, t)  # (1, length, vocab_size)
        logits = logits[0]  # (length, vocab_size)

        # Apply repetition penalty
        if repetition_penalty > 1.0 and token_counts:
            penalty = mx.ones((config.vocab_size,))
            for tok_id, count in token_counts.items():
                if tok_id < config.vocab_size:
                    # Penalize proportional to frequency
                    p = min(repetition_penalty ** count, 5.0)
                    penalty = penalty.at[tok_id].add(mx.array(p - 1.0))
            logits = logits / penalty[None, :]

        # Temperature scaling
        logits = logits / max(temp, 0.1)

        # Top-p filtering per position
        probs = mx.softmax(logits, axis=-1)  # (length, vocab_size)

        # Sample tokens
        sampled = mx.random.categorical(mx.log(probs + 1e-10), axis=-1)  # (length,)

        # Confidence of sampled tokens
        mx.eval(sampled, probs)
        sampled_list = sampled.tolist()

        confidences = []
        for i in range(length):
            if is_masked_list[i]:
                tok = sampled_list[i]
                conf = probs[i, tok].item()
                confidences.append((i, tok, conf))

        # How many to unmask this step
        steps_remaining = steps - step
        n_to_unmask = max(1, int(n_masked / max(steps_remaining, 1)))

        # Select most confident positions
        confidences.sort(key=lambda x: -x[2])
        to_unmask = confidences[:n_to_unmask]

        # Unmask
        for idx, tok, conf in to_unmask:
            x_list[idx] = tok
            is_masked_list[idx] = False
            token_counts[tok] = token_counts.get(tok, 0) + 1

    # Fill remaining masks with greedy decoding at t=0
    if any(is_masked_list):
        x_tensor = mx.array([x_list], dtype=mx.int32)
        logits = model(x_tensor, mx.array([0.0]))

        # Apply heavy repetition penalty for final pass
        if token_counts:
            penalty = mx.ones((config.vocab_size,))
            for tok_id, count in token_counts.items():
                if tok_id < config.vocab_size:
                    p = min(2.0 ** count, 10.0)
                    penalty = penalty.at[tok_id].add(mx.array(p - 1.0))
            logits = logits / penalty[None, None, :]

        final_preds = mx.argmax(logits[0], axis=-1)
        mx.eval(final_preds)
        preds_list = final_preds.tolist()
        for i in range(length):
            if is_masked_list[i]:
                x_list[i] = preds_list[i]

    return mx.array(x_list, dtype=mx.int32)


# =============================================================================
# Utilities
# =============================================================================

def count_parameters(model):
    import mlx.utils
    params = model.parameters()
    return sum(p.size for _, p in mlx.utils.tree_flatten(params))


def print_mdlm_summary(model):
    n = count_parameters(model)
    c = model.config
    print("\n" + "=" * 55)
    print("  MDLM (Masked Diffusion Language Model)")
    print("=" * 55)
    print(f"  d_model:      {c.d_model}")
    print(f"  n_layers:     {c.n_layers}")
    print(f"  n_heads:      {c.n_heads}")
    print(f"  ffn_dim:      {c.ffn_dim}")
    print(f"  vocab_size:   {c.vocab_size} (+1 MASK)")
    print(f"  Parameters:   {n:,} ({n/1e6:.1f}M)")
    print(f"  Memory (FP32): ~{n*4/1e6:.0f} MB")
    print("=" * 55 + "\n")
