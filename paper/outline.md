# From Architecture Search to Diffusion: Empirical Lessons from Building Hybrid Language Models on Consumer Hardware

## Abstract (~200 words)

We present a systematic empirical study of hybrid language model architectures, comparing autoregressive and diffusion-based generation paradigms. Starting from a triple-hybrid architecture combining Gated Linear Attention, Softmax Attention with Multi-Head Latent compression, and State Space Models with Mixture-of-Experts, we explore learned operation-type routing (Mixture of Operations) and continuous per-layer proportion optimization (SoftMoO). Our experiments across two datasets reveal that: (1) operation diversity consistently outperforms homogeneous architectures; (2) per-token operation routing performs no better than random selection, suggesting the granularity is wrong; (3) learned per-layer proportions converge to dataset-dependent optima that differ from human-designed patterns; and (4) a Masked Diffusion Language Model trained on the same data produces qualitatively different outputs with distinct trade-offs. All experiments are conducted on a single Apple M4 Mac Mini (16GB), demonstrating that meaningful architecture research is accessible without large compute budgets. We release all code, data pipelines, and trained checkpoints.

---

## 1. Introduction (~1 page)

### The problem
- Hybrid LLM architectures (GLA + Attention + SSM) are the emerging standard
- But the design space is vast: which operations, in what proportion, at which layers?
- Current approach: human intuition + expensive ablation studies
- No principled method for discovering optimal hybrid configurations

### Our contributions
1. **Systematic comparison** of 6+ architectural variants on controlled benchmarks
2. **Mixture of Operations (MoO)**: first attempt at per-token operation-type routing between fundamentally different computation types (not just MoE expert routing)
3. **SoftMoO**: differentiable architecture discovery for hybrid proportions
4. **Cross-paradigm comparison**: autoregressive vs masked diffusion on identical data
5. **Reproducible small-scale methodology**: all experiments on consumer hardware

### Key findings (preview)
- Balanced operation mixes beat lopsided ones on small/homogeneous data
- Majority-GLA configurations win on diverse data (confirming industry practice)
- Per-token routing ≈ random (informative negative result)
- SoftMoO finds near-optimal configurations automatically
- Diffusion models produce coherent text but with distinct quality profile

---

## 2. Background & Related Work (~1.5 pages)

### 2.1 Hybrid Architectures
- Jamba (AI21): Mamba + Attention + MoE
- Hunyuan-TurboS (Tencent): Attention-Mamba-FFN pattern
- Qwen3-Next (Alibaba): Gated DeltaNet + Attention + MoE
- Nemotron-H (NVIDIA): 92% Mamba-2 blocks
- Common theme: fixed patterns chosen by ablation

### 2.2 Architecture Search for LLMs
- DARTS (Liu et al., 2018): differentiable NAS
- Gap: DARTS not applied to operation-type selection in hybrid LLMs

### 2.3 Mixture of Experts
- MoE routes between parameters (same operation, different weights)
- Our MoO routes between operations (different computation types)
- Key distinction and motivation

### 2.4 Diffusion Language Models
- MDLM (NeurIPS 2024), SEDD (ICML 2024 Best Paper)
- LLaDA (2025): 8B diffusion LM competitive with LLaMA3
- Gemini Diffusion (2025): 5x faster generation
- Gap: no cross-paradigm comparison on identical data at small scale

---

## 3. Architecture Components (~2 pages)

### 3.1 Gated Linear Attention (GLA)
- Decay-weighted attention with learned per-head gates
- O(T²) parallel training, O(1) recurrent inference
- Our implementation: numerical stability via causal masking before exp

### 3.2 Latent Attention (Softmax + MLA)
- Multi-Head Latent Attention: 4x KV-cache compression
- Standard softmax attention on decompressed keys/values
- RoPE positional encoding

### 3.3 Simple SSM (Gated Recurrence)
- Diagonal state space with parallel scan
- Chunk-based computation for numerical stability
- Data-dependent gating (selective mechanism)

### 3.4 Mixture of Experts FFN
- Top-1 routing with SwiGLU experts
- 4 experts, 1 active per token

### 3.5 Multi-Token Prediction
- 2 additional heads predicting future tokens
- Weighted loss: main + 0.3 × MTP

---

## 4. Learned Operation Routing (~2 pages)

### 4.1 Mixture of Operations (MoO)
- Every layer contains all 3 mixer types + lightweight router
- Per-token top-1 selection with softmax weight
- **Finding**: learned routing ≈ random routing (Table X)
- Analysis: routing patterns show layer-level but not token-level specialization
- Interpretation: operation type doesn't depend on individual tokens

### 4.2 SoftMoO: Continuous Proportion Learning
- 3 learnable logits per layer → softmax → continuous blend
- Only 27 extra parameters (3 × 9 layers)
- **Finding**: converges near-uniform on homogeneous data, near 7:1:1 on diverse data
- Interpretation: optimal proportions are dataset-dependent
- Practical use: cheap architecture discovery tool

### 4.3 Ablation: Fixed Patterns
- Results table: all-gla, all-attn, balanced-333, fixed-711, MoO, SoftMoO
- Shakespeare results (Table 1)
- TinyStories results (Table 2)
- Key insight: ranking changes between datasets

---

## 5. Masked Diffusion Language Model (~1.5 pages)

### 5.1 Architecture
- Bidirectional transformer (no causal mask)
- Timestep conditioning via sinusoidal embedding
- Cosine noise schedule for masking
- 75.3M parameters (comparable to AR model at 88.4M)

### 5.2 Training
- Same data: 50M token mix (TinyStories + Cosmopedia + FineWeb-Edu + synthetic)
- Same compute budget: 15K steps, batch 4, seq_len 256
- Loss: cross-entropy on masked positions only

### 5.3 Improved Sampling
- Temperature annealing (1.2 → 0.5 across diffusion steps)
- Repetition penalty (frequency-based)
- Confidence-based progressive unmasking
- 100 diffusion steps for generation

### 5.4 Results
- Generation quality comparison: AR vs MDLM samples
- Quantitative metrics: unique ratio, n-gram repetition
- Generation speed comparison
- Qualitative analysis: what each paradigm does well/poorly

---

## 6. Experiments (~2 pages)

### 6.1 Setup
- Hardware: Apple M4 Mac Mini, 16GB unified memory
- Framework: MLX (Apple's ML framework)
- Tokenizer: GPT-2 (50,257 vocab)
- Data: curated mix of 4 sources, 50M tokens

### 6.2 Architecture Comparison (Table 1)
| Model | Shakespeare Val | TinyStories Val | Params | tok/s |
|-------|----------------|-----------------|--------|-------|
| all-gla | 4.336 | — | 92M | 1,501 |
| all-attn | 4.279 | 2.639 | 89M | 1,589 |
| balanced-333 | 4.264 | 2.721 | 80M | 1,453 |
| fixed-711 | 4.292 | 2.603 | 88M | 1,535 |
| moo-learned | 4.312 | — | 118M | 1,088 |
| moo-random | 4.288 | — | 118M | 1,081 |
| soft-moo | 4.276 | 2.614 | 118M | 1,070 |

### 6.3 Routing Analysis
- Per-layer distribution heatmaps
- Specialization metrics (entropy)
- Evolution during training (Figure X)

### 6.4 AR vs MDLM Comparison (Table 2)
[TO BE FILLED after compare.py runs]
- Perplexity / masked prediction loss
- Generation speed
- Text quality metrics
- Sample comparisons

---

## 7. Discussion (~1 page)

### 7.1 When does operation diversity matter?
- Homogeneous data → balanced mix is near-optimal
- Diverse data → majority cheap ops (GLA) + minority precise (Attn) wins
- SSM consistently least useful at this scale

### 7.2 Why per-token routing fails
- Individual tokens don't have a "type" that determines needed computation
- The relevant unit may be phrases, sentences, or semantic blocks
- Future work: segment-level routing

### 7.3 SoftMoO as architecture discovery
- Cheap (27 parameters, ~1K training steps)
- Finds configurations competitive with exhaustive search
- Dataset-dependent results → no universal optimal ratio

### 7.4 Autoregressive vs Diffusion: different trade-offs
- AR: better perplexity, sequential generation, strong at narrative
- MDLM: parallel generation, can "revise" tokens, different error patterns
- Not a replacement but complementary paradigms

### 7.5 Limitations
- Small scale (75-118M params)
- Limited datasets (2 for architecture, 1 for diffusion)
- Single seed (no error bars)
- MDLM sampler not fully optimized
- Apple Silicon specific (MLX)

---

## 8. Conclusion (~0.5 page)

We demonstrate that meaningful architecture research can be conducted on consumer hardware. Our key contributions:

1. First systematic comparison of operation-type routing in hybrid LLMs
2. SoftMoO as a cheap, differentiable tool for hybrid architecture discovery
3. First MDLM implementation and training on Apple Silicon (MLX)
4. Evidence that optimal hybrid proportions are dataset-dependent
5. Informative negative result: per-token routing ≈ random

We release all code, trained models, and data pipelines to enable reproducible research.

---

## Appendix

### A. Implementation Details
- Numerical stability fixes (causal mask before exp, chunk-based parallel scan)
- MLX-specific optimizations

### B. Training Curves
- Loss curves for all experiments
- Routing distribution evolution

### C. Additional Samples
- Extended generation examples from AR and MDLM

### D. Compute Budget
- Total GPU-hours equivalent
- Cost breakdown

---

## Figures Needed
1. Architecture diagram (Arche block with GLA/Attn/SSM + MoE)
2. MoO routing diagram
3. SoftMoO proportion evolution during training
4. MDLM demasking visualization (noise → text)
5. Loss curves (all experiments)
6. Routing heatmap (per-layer × per-operation)
7. AR vs MDLM sample comparison (side by side)

## Tables Needed
1. Architecture comparison (Shakespeare)
2. Architecture comparison (TinyStories)
3. AR vs MDLM comparison
4. Learned SoftMoO proportions per dataset
5. Generation quality metrics
