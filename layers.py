"""
Arche Architecture - Layer Implementations
===========================================
Triple-hybrid mixer layers + MoE FFN for the Arche architecture.

Layers:
  1. GatedLinearAttention    — O(T²) training (parallel), O(1) inference (recurrent)
  2. LatentAttention         — Softmax attention with MLA-style KV compression
  3. SimpleSSM               — Diagonal gated recurrence with parallel scan
  4. MoEFFN                  — Top-1 Mixture-of-Experts feed-forward
  5. DenseFFN                — Standard SwiGLU feed-forward
  6. MixtureOfOperations     — Routes tokens to different operation TYPES (the innovation)
  7. ArcheBlock              — Pre-norm residual block (mixer + FFN)
"""

import mlx.core as mx
import mlx.nn as nn


def silu(x):
    """SiLU / Swish activation: x * sigmoid(x)"""
    return x * mx.sigmoid(x)


# =============================================================================
# 1. Gated Linear Attention (GLA)
# =============================================================================
# Core idea: attention with learnable exponential decay instead of softmax.
# - Training: parallel O(T²) via decay-weighted attention matrix
# - Inference: recurrent O(1) per token via state matrix update
#
# State update:  S_t = α_t · S_{t-1} + (β_t ⊙ k_t) ⊗ v_t
# Output:        o_t = q_t^T · S_t
#
# Where α is a per-head decay gate and β is a per-head update gate.
# =============================================================================

class GatedLinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        inner_dim = config.n_heads * config.head_dim

        self.wq = nn.Linear(config.d_model, inner_dim, bias=False)
        self.wk = nn.Linear(config.d_model, inner_dim, bias=False)
        self.wv = nn.Linear(config.d_model, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, config.d_model, bias=False)

        # Per-head scalar gates (data-dependent)
        self.w_alpha = nn.Linear(config.d_model, config.n_heads, bias=True)  # decay
        self.w_beta = nn.Linear(config.d_model, config.n_heads, bias=True)   # update

    def __call__(self, x, cache=None):
        B, T, _ = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # All: (B, n_heads, T, head_dim)

        # Compute data-dependent gates
        alpha = mx.sigmoid(self.w_alpha(x))  # (B, T, n_heads) — decay
        beta = mx.sigmoid(self.w_beta(x))    # (B, T, n_heads) — update

        alpha = alpha.transpose(0, 2, 1)  # (B, n_heads, T)
        beta = beta.transpose(0, 2, 1)    # (B, n_heads, T)

        # Scale keys by update gate
        k = k * beta[..., None]  # (B, n_heads, T, head_dim)

        # --- Recurrent mode (inference: one token at a time) ---
        if cache is not None:
            state = cache.get("gla_state", mx.zeros((B, self.n_heads, self.head_dim, self.head_dim)))
            a = alpha[:, :, :, None, None]   # (B, H, T=1, 1, 1)
            # State update: S = α·S + k^T @ v (rank-1 outer product update)
            new_state = a[:, :, 0] * state + k.transpose(0, 1, 3, 2) @ v
            output = (q @ new_state).transpose(0, 2, 1, 3).reshape(B, T, -1)
            cache["gla_state"] = new_state
            return self.wo(output)

        # --- Parallel mode (training: full sequence) ---
        # Compute cumulative log-decay for the T×T decay matrix
        log_alpha = mx.log(mx.clip(alpha, a_min=1e-6, a_max=1.0))
        cum_log = mx.cumsum(log_alpha, axis=-1)  # (B, n_heads, T)

        # decay[i,j] = exp(cum_log[i] - cum_log[j]) for causal positions (i >= j)
        diff = cum_log[:, :, :, None] - cum_log[:, :, None, :]  # (B, H, T, T)
        # CRITICAL: mask non-causal positions BEFORE exp to avoid inf * 0 = NaN
        causal_mask = mx.tril(mx.ones((T, T))).astype(mx.bool_)
        diff = mx.where(causal_mask, diff, mx.array(-1e9))
        diff = mx.clip(diff, a_min=-80.0, a_max=0.0)  # Prevent exp overflow
        decay = mx.exp(diff)

        # Decayed attention: scores = (Q @ K^T) * decay / √d
        scores = (q @ k.transpose(0, 1, 3, 2)) * decay / (self.head_dim ** 0.5)
        output = scores @ v  # (B, n_heads, T, head_dim)

        output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(output)


# =============================================================================
# 2. Latent Attention (Softmax + MLA)
# =============================================================================
# Standard softmax attention but with Multi-Head Latent Attention compression:
# - Keys and Values are jointly compressed into a shared latent vector
# - At inference, cache the latent (4x smaller than full KV)
# - RoPE applied for positional encoding (only in this layer type)
# =============================================================================

class LatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        inner_dim = config.n_heads * config.head_dim

        # Q projection (standard)
        self.wq = nn.Linear(config.d_model, inner_dim, bias=False)

        # MLA: compress → latent → decompress K, V
        self.kv_compress = nn.Linear(config.d_model, config.mla_latent_dim, bias=False)
        self.k_decompress = nn.Linear(config.mla_latent_dim, inner_dim, bias=False)
        self.v_decompress = nn.Linear(config.mla_latent_dim, inner_dim, bias=False)

        self.wo = nn.Linear(inner_dim, config.d_model, bias=False)
        self.rope = nn.RoPE(config.head_dim, base=config.rope_theta)

    def __call__(self, x, cache=None):
        B, T, _ = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # MLA compression: x → latent → K, V
        latent = self.kv_compress(x)  # (B, T, latent_dim) — cached at inference

        # Handle KV cache (stores compressed latent)
        if cache is not None:
            prev = cache.get("mla_latent", None)
            if prev is not None:
                latent_full = mx.concatenate([prev, latent], axis=1)
            else:
                latent_full = latent
            cache["mla_latent"] = latent_full
        else:
            latent_full = latent

        # Decompress to full K, V
        T_k = latent_full.shape[1]
        k = self.k_decompress(latent_full).reshape(B, T_k, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_decompress(latent_full).reshape(B, T_k, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE positional encoding
        offset = T_k - T
        q = self.rope(q, offset=offset)
        k = self.rope(k)

        # Scaled dot-product attention with causal mask
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if T > 1:
            # Causal mask: -inf for future positions
            mask = mx.triu(mx.full((T, T_k), -1e9), k=T_k - T + 1)
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        output = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(output)


# =============================================================================
# 3. Simple SSM (Gated Recurrence with Parallel Scan)
# =============================================================================
# Simplified state space model using diagonal gated recurrence:
#   h_t = gate_t · h_{t-1} + (1 - gate_t) · value_t
#   y_t = out_proj(h_t ⊙ output_gate_t)
#
# Training uses the closed-form parallel scan:
#   h_t = exp(Σ log(gate)) · cumsum(exp(-Σ log(gate)) · input)
#
# This captures sequential patterns (counting, state machines) that
# linear attention struggles with.
# =============================================================================

class SimpleSSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        inner = d * config.ssm_expand

        self.in_proj = nn.Linear(d, inner, bias=False)     # value path
        self.gate_proj = nn.Linear(d, inner, bias=True)     # forget gate
        self.z_proj = nn.Linear(d, inner, bias=False)       # output gate
        self.out_proj = nn.Linear(inner, d, bias=False)

    def __call__(self, x, cache=None):
        B, T, D = x.shape

        v = silu(self.in_proj(x))            # (B, T, inner) — value to store
        gate = mx.sigmoid(self.gate_proj(x)) # (B, T, inner) — forget/decay gate
        z = silu(self.z_proj(x))             # (B, T, inner) — output gate

        # --- Recurrent mode (inference) ---
        if cache is not None:
            h_prev = cache.get("ssm_h", mx.zeros_like(v[:, 0:1]))
            h = gate * h_prev + (1 - gate) * v
            cache["ssm_h"] = h
            return self.out_proj(h * z)

        # --- Parallel scan (training) ---
        # Use chunk-based scan for numerical stability.
        # The closed-form exp/cumsum can overflow for long sequences,
        # so we split into chunks and carry state across them.
        inp = (1 - gate) * v  # (B, T, inner)
        chunk_size = 32
        h = mx.zeros_like(inp)
        h_prev = mx.zeros((B, 1, inp.shape[-1]))

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            g_chunk = gate[:, start:end]         # (B, C, inner)
            inp_chunk = inp[:, start:end]        # (B, C, inner)

            log_g = mx.log(mx.clip(g_chunk, a_min=1e-6, a_max=1.0))
            cum_log = mx.cumsum(log_g, axis=1)   # (B, C, inner)
            cum_log = mx.clip(cum_log, a_min=-40.0, a_max=0.0)

            # Contribution from previous chunks: decayed h_prev
            carry = mx.exp(cum_log) * h_prev     # (B, C, inner)

            # Contribution from current chunk (parallel scan within chunk)
            weighted = mx.exp(-cum_log) * inp_chunk
            cum_weighted = mx.cumsum(weighted, axis=1)
            local = mx.exp(cum_log) * cum_weighted

            h_chunk = carry + local
            h = h.at[:, start:end].add(h_chunk)

            # Carry the last hidden state to the next chunk
            h_prev = h_chunk[:, -1:]

        return self.out_proj(h * z)


# =============================================================================
# 4. Mixture of Experts FFN
# =============================================================================
# Top-1 routing with SwiGLU experts.
# Only 1 of N experts activates per token → N× fewer FLOPs than dense.
#
# Uses auxiliary-loss-free load balancing concept from DeepSeek-V3:
# instead of a balancing loss term, we could add dynamic bias to router logits.
# (Simplified for this PoC — full balancing can be added later.)
# =============================================================================

class Expert(nn.Module):
    """Single SwiGLU expert."""
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)   # up projection
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)    # down projection
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)    # gate projection

    def __call__(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.experts = [Expert(config.d_model, config.ffn_dim) for _ in range(config.n_experts)]
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)

    def __call__(self, x):
        B, T, D = x.shape

        # Router: compute expert selection logits
        logits = self.router(x)                        # (B, T, n_experts)
        probs = mx.softmax(logits, axis=-1)            # (B, T, n_experts)
        indices = mx.argmax(logits, axis=-1)           # (B, T)

        # Compute all expert outputs and select via mask
        # (For small n_experts this is efficient; production would use scatter/gather)
        output = mx.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).astype(x.dtype)      # (B, T) — 1 where expert i is selected
            weight = probs[:, :, i]                    # (B, T) — softmax probability
            expert_out = expert(x)                     # (B, T, D)
            output = output + (mask * weight)[..., None] * expert_out

        return output


# =============================================================================
# 5. Dense FFN (for SSM layers — stability)
# =============================================================================

class DenseFFN(nn.Module):
    """Standard SwiGLU FFN without MoE (used in SSM blocks for stability)."""
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.ffn_dim, bias=False)

    def __call__(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


# =============================================================================
# 6. Compressed Attention (CompressedAttn) — INTRA-LAYER COMPOSITION
# =============================================================================
# The key idea: instead of choosing BETWEEN GLA and Attention, COMPOSE them.
#
# GLA compresses the sequence (T → T/r), Attention operates on the compressed
# version (precise but cheap), then we expand back (T/r → T).
#
# Cost:  O(T·D) for compression + O((T/r)²·D) for attention + O(T·D) for expand
#        ≈ O(T·D) when r is large enough — SUBQUADRATIC in T
# Quality: Near full attention, because attention sees compressed context
#
# This is a NEW PRIMITIVE — not stacking layers, but composing operations
# within a single layer. Like an hourglass inside each block.
#
# Nobody has done intra-layer composition of different operation types.
# =============================================================================

class CompressedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        inner_dim = config.n_heads * config.head_dim
        self.compress_ratio = getattr(config, "compress_ratio", 4)

        # --- Stage 1: Compress (learned downsampling via strided conv-like projection) ---
        # Maps T tokens → T/r "summary" tokens using learned weighted pooling
        self.compress_gate = nn.Linear(config.d_model, 1, bias=True)   # per-token importance
        self.compress_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # --- Stage 2: Precise attention on compressed sequence ---
        self.wq = nn.Linear(config.d_model, inner_dim, bias=False)
        self.wk = nn.Linear(config.d_model, inner_dim, bias=False)
        self.wv = nn.Linear(config.d_model, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, config.d_model, bias=False)
        self.rope = nn.RoPE(config.head_dim, base=config.rope_theta)

        # --- Stage 3: Expand back (cross-attention from full seq to compressed) ---
        self.expand_q = nn.Linear(config.d_model, inner_dim, bias=False)  # queries from original positions
        self.expand_k = nn.Linear(config.d_model, inner_dim, bias=False)  # keys from compressed
        self.expand_v = nn.Linear(config.d_model, inner_dim, bias=False)  # values from compressed

    def _compress(self, x):
        """Compress T tokens → T/r summary tokens using learned weighted pooling."""
        B, T, D = x.shape
        r = self.compress_ratio

        # Pad T to be divisible by r
        pad = (r - T % r) % r
        if pad > 0:
            x = mx.concatenate([x, mx.zeros((B, pad, D))], axis=1)
        T_padded = x.shape[1]
        n_groups = T_padded // r

        # Reshape into groups of r tokens: (B, n_groups, r, D)
        x_groups = x.reshape(B, n_groups, r, D)

        # Learned importance weights per token within each group
        gates = mx.sigmoid(self.compress_gate(x_groups))  # (B, n_groups, r, 1)
        gates = gates / (mx.sum(gates, axis=2, keepdims=True) + 1e-6)  # normalize within group

        # Weighted sum within each group → (B, n_groups, D)
        compressed = mx.sum(gates * x_groups, axis=2)
        compressed = self.compress_proj(compressed)

        return compressed, T  # Return original T for unpadded expansion

    def _attend(self, compressed, cache=None):
        """Standard causal attention on the compressed sequence."""
        B, T_c, _ = compressed.shape

        q = self.wq(compressed).reshape(B, T_c, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(compressed).reshape(B, T_c, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(compressed).reshape(B, T_c, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if T_c > 1:
            mask = mx.triu(mx.full((T_c, T_c), -1e9), k=1)
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T_c, -1)
        return self.wo(out)

    def _expand(self, x_original, compressed_attended):
        """Expand T/r → T via cross-attention (original queries, compressed KV)."""
        B, T, D = x_original.shape
        T_c = compressed_attended.shape[1]

        # Queries from original token positions
        q = self.expand_q(x_original).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        # Keys and values from compressed attended output
        k = self.expand_k(compressed_attended).reshape(B, T_c, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.expand_v(compressed_attended).reshape(B, T_c, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Cross-attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Causal mask: token at position i can only attend to compressed group j where j <= i/r
        r = self.compress_ratio
        row_idx = mx.arange(T)[:, None]       # (T, 1)
        col_idx = mx.arange(T_c)[None, :]     # (1, T_c)
        causal_mask = mx.where(col_idx <= row_idx / r, 0.0, -1e9)  # (T, T_c)
        scores = scores + causal_mask

        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return out

    def __call__(self, x, cache=None):
        B, T, D = x.shape

        # Stage 1: Compress T → T/r
        compressed, T_orig = self._compress(x)  # (B, T/r, D)

        # Stage 2: Self-attention on compressed (cheap: O((T/r)²))
        attended = self._attend(compressed, cache)  # (B, T/r, D)

        # Stage 3: Expand back T/r → T via cross-attention
        output = self._expand(x[:, :T_orig], attended)  # (B, T, D)

        return output


# =============================================================================
# 7. Mixture of Operations (MoO) — PER-TOKEN ROUTING
# =============================================================================
# Instead of a fixed layer type per block, MoO routes each token to the
# best operation TYPE: linear attention, softmax attention, or SSM.
#
# This is fundamentally different from MoE:
#   MoE  = same operation, different parameters (which expert?)
#   MoO  = different operations, same routing principle (which computation?)
#
# No published work routes between fundamentally different layer types
# on a per-token basis. This tests the hypothesis that different tokens
# need different KINDS of computation, not just different parameters.
# =============================================================================

class MixtureOfOperations(nn.Module):
    OP_NAMES = ["gla", "attn", "ssm"]

    def __init__(self, config):
        super().__init__()
        # All 3 mixer types live in every layer
        self.gla = GatedLinearAttention(config)
        self.attn = LatentAttention(config)
        self.ssm = SimpleSSM(config)

        # Lightweight router: d_model → 3 logits
        self.router = nn.Linear(config.d_model, 3, bias=False)
        self.random_routing = getattr(config, "moo_random_routing", False)

        # Stored after each forward for analysis (not part of grad graph)
        self._last_probs = None
        self._last_indices = None

    def __call__(self, x, cache=None):
        B, T, D = x.shape

        # --- Router: per-token operation selection ---
        if self.random_routing:
            # Ablation: random routing (no learned specialization)
            indices = mx.random.randint(0, 3, (B, T))
            probs = mx.ones((B, T, 3)) / 3.0
        else:
            logits = self.router(x)                     # (B, T, 3)
            probs = mx.softmax(logits, axis=-1)         # (B, T, 3)
            indices = mx.argmax(logits, axis=-1)         # (B, T)

        # Store for analysis
        self._last_probs = probs
        self._last_indices = indices

        # --- Compute all 3 mixer outputs ---
        # (all mixers run on all tokens; production would use conditional compute)
        gla_out = self.gla(x, cache=cache)
        attn_out = self.attn(x, cache=cache)
        ssm_out = self.ssm(x, cache=cache)

        # --- Top-1 selection with softmax weight (gradient flows through prob) ---
        outputs = [gla_out, attn_out, ssm_out]
        result = mx.zeros_like(x)
        for i, out in enumerate(outputs):
            mask = (indices == i).astype(x.dtype)[..., None]   # (B, T, 1)
            weight = probs[:, :, i:i + 1]                     # (B, T, 1)
            result = result + mask * weight * out

        return result


# =============================================================================
# 7. Soft Mixture of Operations (SoftMoO) — LEARNED PER-LAYER PROPORTIONS
# =============================================================================
# Instead of per-token routing (which didn't beat random), learn a continuous
# weighted blend of operations at each layer. Only 3 parameters per layer.
#
# This answers the question: "what is the OPTIMAL PROPORTION of each operation
# at each layer?" — producing a prescriptive result others can use.
#
# E.g., Layer 3 = 72% Attn + 18% GLA + 10% SSM
# =============================================================================

class SoftMixtureOfOperations(nn.Module):
    OP_NAMES = ["gla", "attn", "ssm"]

    def __init__(self, config):
        super().__init__()
        self.gla = GatedLinearAttention(config)
        self.attn = LatentAttention(config)
        self.ssm = SimpleSSM(config)

        # 3 learnable logits → softmax → proportions
        # Initialized to 0 → uniform [33%, 33%, 33%]
        self.mix_logits = mx.zeros((3,))

    def get_proportions(self):
        """Return current mix proportions as (3,) array."""
        return mx.softmax(self.mix_logits)

    def __call__(self, x, cache=None):
        w = mx.softmax(self.mix_logits)  # (3,)

        gla_out = self.gla(x, cache=cache)
        attn_out = self.attn(x, cache=cache)
        ssm_out = self.ssm(x, cache=cache)

        return w[0] * gla_out + w[1] * attn_out + w[2] * ssm_out


# =============================================================================
# 8. ArcheBlock (Pre-Norm Residual)
# =============================================================================

class ArcheBlock(nn.Module):
    """One block: RMSNorm → Mixer → Residual → RMSNorm → FFN → Residual"""
    def __init__(self, mixer, ffn, d_model):
        super().__init__()
        self.mixer = mixer
        self.ffn = ffn
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def __call__(self, x, cache=None):
        x = x + self.mixer(self.norm1(x), cache=cache)
        x = x + self.ffn(self.norm2(x))
        return x
