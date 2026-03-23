"""
Arche Model Configuration
=========================
Triple-hybrid architecture: Gated Linear Attention + Latent Softmax Attention + SSM
with Mixture-of-Experts FFN and Multi-Token Prediction.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ArcheConfig:
    # ---------- Model Dimensions ----------
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64          # d_model // n_heads
    ffn_dim: int = 1024         # MoE expert intermediate size

    # ---------- Layer Configuration ----------
    n_layers: int = 9
    # Pattern options:
    #   "gla"  = fixed Gated Linear Attention
    #   "attn" = fixed Softmax Attention + MLA
    #   "ssm"  = fixed Simple SSM
    #   "moo"  = Mixture of Operations (learned routing between all 3)
    layer_pattern: Optional[List[str]] = None
    use_moo: bool = False  # If True, all layers become MoO (overrides layer_pattern)
    moo_random_routing: bool = False  # Ablation: random routing instead of learned
    use_soft_moo: bool = False  # If True, all layers use SoftMoO (learned proportions)

    # ---------- Mixture of Experts ----------
    n_experts: int = 4
    n_active_experts: int = 1   # Top-1 routing (most efficient)

    # ---------- Multi-Head Latent Attention ----------
    mla_latent_dim: int = 128   # 4x compression vs full KV (512 -> 128)

    # ---------- State Space Model ----------
    ssm_expand: int = 2         # Inner dimension = d_model * ssm_expand

    # ---------- Compressed Attention ----------
    compress_ratio: int = 4     # Compress T → T/4 before attention

    # ---------- Multi-Token Prediction ----------
    mtp_num_future: int = 2     # Predict 2 extra future tokens
    mtp_weight: float = 0.3     # Weight of MTP loss

    # ---------- Vocabulary & Sequence ----------
    vocab_size: int = 50257     # GPT-2 tokenizer
    max_seq_len: int = 1024

    # ---------- RoPE (only for softmax attention layers) ----------
    rope_theta: float = 10000.0

    # ---------- Training Defaults ----------
    batch_size: int = 4
    seq_len: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    eval_interval: int = 250
    save_interval: int = 500
    log_interval: int = 10

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.head_dim, \
            f"d_model ({self.d_model}) must equal n_heads * head_dim ({self.n_heads * self.head_dim})"

        if self.use_soft_moo:
            self.layer_pattern = ["soft_moo"] * self.n_layers
        elif self.use_moo:
            self.layer_pattern = ["moo"] * self.n_layers
        elif self.layer_pattern is None:
            # Default 7:1:1 triple-hybrid pattern
            self.layer_pattern = [
                "gla", "gla", "gla",    # Bulk processing
                "attn",                  # Precise retrieval (MLA)
                "gla", "gla",            # More processing
                "ssm",                   # State tracking / sequential patterns
                "gla", "gla",            # Final processing
            ]

        assert len(self.layer_pattern) == self.n_layers, \
            f"layer_pattern length ({len(self.layer_pattern)}) must match n_layers ({self.n_layers})"

    def param_summary(self):
        """Estimate parameter counts."""
        inner = self.n_heads * self.head_dim
        # Embedding
        embed = self.vocab_size * self.d_model
        # Per GLA layer
        gla_mixer = 4 * self.d_model * inner + 2 * self.d_model  # QKVO + gates
        # Per Attention+MLA layer
        attn_mixer = (self.d_model * inner                        # Q
                      + self.d_model * self.mla_latent_dim        # KV compress
                      + self.mla_latent_dim * inner * 2           # K,V decompress
                      + inner * self.d_model)                     # O
        # Per SSM layer
        ssm_mixer = 4 * self.d_model * (self.d_model * self.ssm_expand)  # projections
        # MoE FFN
        moe_ffn = self.n_experts * 3 * self.d_model * self.ffn_dim + self.d_model * self.n_experts
        # Dense FFN
        dense_ffn = 3 * self.d_model * self.ffn_dim
        # MTP
        mtp = self.mtp_num_future * self.d_model * self.d_model

        n_gla = self.layer_pattern.count("gla")
        n_attn = self.layer_pattern.count("attn")
        n_ssm = self.layer_pattern.count("ssm")

        total = (embed
                 + n_gla * (gla_mixer + moe_ffn)
                 + n_attn * (attn_mixer + moe_ffn)
                 + n_ssm * (ssm_mixer + dense_ffn)
                 + mtp
                 + self.n_layers * 2 * self.d_model)  # norms

        active = total - (n_gla + n_attn) * (self.n_experts - self.n_active_experts) * 3 * self.d_model * self.ffn_dim

        return {
            "total_params": total,
            "active_params": active,
            "total_mb": total * 4 / 1e6,  # FP32
            "active_mb": active * 4 / 1e6,
        }
