"""
Arche Model
============
Composes the triple-hybrid architecture:
  - Embedding → [ArcheBlocks × N] → RMSNorm → LM Head + MTP Heads

The model uses weight tying (embedding weights = output projection)
and Multi-Token Prediction heads for improved training efficiency
and built-in speculative decoding at inference.
"""

import mlx.core as mx
import mlx.nn as nn

from config import ArcheConfig
from layers import (
    GatedLinearAttention,
    LatentAttention,
    SimpleSSM,
    MoEFFN,
    DenseFFN,
    CompressedAttention,
    MixtureOfOperations,
    SoftMixtureOfOperations,
    ArcheBlock,
    silu,
)


class MTPHead(nn.Module):
    """
    Multi-Token Prediction heads.
    Each head learns to predict a different future token position.
    Head k predicts token at position t+k+2 from hidden state at position t.
    (Head 0 → t+2, Head 1 → t+3, etc.)

    At inference, these serve as a built-in "draft model" for speculative decoding
    without needing a separate model.
    """

    def __init__(self, config):
        super().__init__()
        self.n_future = config.mtp_num_future
        self.projections = [
            nn.Linear(config.d_model, config.d_model, bias=False)
            for _ in range(config.mtp_num_future)
        ]

    def __call__(self, hidden_states, embed_weight):
        """
        Args:
            hidden_states: (B, T, d_model)
            embed_weight: (vocab_size, d_model) — shared output projection

        Returns:
            list of (B, T, vocab_size) logits for each future position
        """
        results = []
        for proj in self.projections:
            h = silu(proj(hidden_states))
            logits = h @ embed_weight.T
            results.append(logits)
        return results


class ArcheModel(nn.Module):
    """
    The complete Arche language model.

    Architecture:
        Embedding → [GLA/Attn/SSM Block × 9] → RMSNorm → LM Head

    Layer pattern (default 7:1:1):
        - 7× Gated Linear Attention + MoE FFN  (bulk processing)
        - 1× Softmax Attention + MLA + MoE FFN (precise retrieval)
        - 1× SSM + Dense FFN                    (sequential patterns)
    """

    def __init__(self, config: ArcheConfig):
        super().__init__()
        self.config = config

        # Token embedding (weights shared with output projection)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Build blocks based on layer pattern
        self.blocks = []
        for layer_type in config.layer_pattern:
            if layer_type == "gla":
                mixer = GatedLinearAttention(config)
                ffn = MoEFFN(config)
            elif layer_type == "attn":
                mixer = LatentAttention(config)
                ffn = MoEFFN(config)
            elif layer_type == "ssm":
                mixer = SimpleSSM(config)
                ffn = DenseFFN(config)
            elif layer_type == "cattn":
                mixer = CompressedAttention(config)
                ffn = MoEFFN(config)
            elif layer_type == "moo":
                mixer = MixtureOfOperations(config)
                ffn = MoEFFN(config)
            elif layer_type == "soft_moo":
                mixer = SoftMixtureOfOperations(config)
                ffn = MoEFFN(config)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            self.blocks.append(ArcheBlock(mixer, ffn, config.d_model))

        # Final normalization
        self.norm = nn.RMSNorm(config.d_model)

        # Multi-Token Prediction heads
        self.mtp = MTPHead(config)

    def __call__(self, tokens, cache=None):
        """
        Args:
            tokens: (B, T) integer token IDs
            cache: optional list of dicts for inference caching

        Returns:
            logits: (B, T, vocab_size) — next-token prediction logits
            mtp_logits: list of (B, T, vocab_size) — future token predictions
        """
        x = self.embed(tokens)  # (B, T, d_model)

        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x = block(x, cache=layer_cache)

        x = self.norm(x)

        # LM head (weight-tied with embedding)
        logits = x @ self.embed.weight.T  # (B, T, vocab_size)

        # Multi-token prediction heads
        mtp_logits = self.mtp(x, self.embed.weight)

        return logits, mtp_logits

    def make_cache(self):
        """Create empty cache dicts for autoregressive generation."""
        return [{} for _ in range(len(self.blocks))]

    def get_learned_proportions(self):
        """Get learned proportions from SoftMoO layers. Returns list of (layer_idx, proportions)."""
        results = []
        for i, block in enumerate(self.blocks):
            if isinstance(block.mixer, SoftMixtureOfOperations):
                props = block.mixer.get_proportions()
                results.append((i, props))
        return results

    def get_routing_stats(self):
        """
        Collect routing statistics from MoO layers after a forward pass.
        Returns list of (layer_idx, counts_array) for each MoO layer.
        """
        stats = []
        for i, block in enumerate(self.blocks):
            if isinstance(block.mixer, MixtureOfOperations) and block.mixer._last_indices is not None:
                indices = block.mixer._last_indices  # (B, T)
                counts = []
                for op in range(3):
                    c = mx.sum((indices == op).astype(mx.float32))
                    counts.append(c)
                counts = mx.stack(counts)
                stats.append((i, counts))
        return stats


def count_parameters(model):
    """Count total and trainable parameters."""
    params = model.parameters()
    import mlx.utils
    total = sum(p.size for name, p in mlx.utils.tree_flatten(params))
    return total


def print_model_summary(model, config):
    """Print a summary of the model architecture."""
    n_params = count_parameters(model)
    est = config.param_summary()

    print("\n" + "=" * 60)
    print("  ARCHE MODEL SUMMARY")
    print("=" * 60)
    print(f"  Layer pattern:    {config.layer_pattern}")
    print(f"  d_model:          {config.d_model}")
    print(f"  n_heads:          {config.n_heads}")
    print(f"  head_dim:         {config.head_dim}")
    print(f"  n_experts:        {config.n_experts} (top-{config.n_active_experts})")
    print(f"  MLA latent dim:   {config.mla_latent_dim}")
    print(f"  SSM expand:       {config.ssm_expand}x")
    print(f"  MTP heads:        {config.mtp_num_future}")
    print(f"  Vocab size:       {config.vocab_size}")
    print(f"  ---")
    print(f"  Total params:     {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Est. active:      ~{est['active_params']:,} ({est['active_params']/1e6:.1f}M)")
    print(f"  Memory (FP32):    ~{n_params * 4 / 1e6:.0f} MB")
    print(f"  Memory (FP16):    ~{n_params * 2 / 1e6:.0f} MB")
    print("=" * 60 + "\n")
