# Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison

This repository contains the code, data pipelines, and evaluation scripts for the paper:

> **Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison**
> Caio Vicentino, 2025

We train an autoregressive (AR) Transformer and a Masked Diffusion Language Model (MDLM) on identical data (50M tokens from TinyStories), identical compute budget (20K steps, batch 32), and identical hardware (NVIDIA H100 80GB), isolating the generation paradigm as the sole variable.

## Key Findings

1. **Training throughput is near-identical**: MDLM trains at 95.5% of AR throughput (48,343 vs. 50,620 tok/s)
2. **Convergence regimes differ**: AR overfits after ~14K steps; MDLM continues improving through 20K steps
3. **Generation profiles differ**: AR produces fluent but formulaic text; MDLM produces diverse but occasionally ungrammatical text

## Results

| Metric | AR | MDLM |
|--------|-----|------|
| Parameters | 123.6M | 162.7M |
| Training time | 107.9 min | 113.0 min |
| Throughput | 50,620 tok/s | 48,343 tok/s |
| Best val loss | 1.589 | 3.412 |

> Val losses use different objectives and are not cross-comparable.

## Repository Structure

```
arche/
├── train.py                  # AR model training
├── train_diffusion.py        # MDLM training
├── model.py                  # AR Transformer architecture
├── mdlm.py                   # Masked Diffusion LM architecture
├── config.py                 # Model configuration
├── layers.py                 # Shared layer implementations
├── data.py                   # Data loading and tokenization
├── generate.py               # Text generation (AR)
├── compare.py                # AR vs MDLM evaluation
├── benchmark.py              # Architecture benchmarking
├── analyze.py                # Results analysis
├── prepare_training_data.py  # Data pipeline
├── requirements.txt          # Dependencies
├── paper/                    # Paper source (LaTeX)
│   ├── paper.tex
│   ├── paper.md
│   ├── fig_loss_curves.pdf
│   └── fig_throughput.pdf
└── gpu/                      # H100 training scripts
```

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train AR model

```bash
python train.py --data data/tinystories.txt --steps 20000 --batch 32 --seq-len 512
```

### Train MDLM

```bash
python train_diffusion.py --data data/tinystories.txt --steps 20000 --batch 32 --seq-len 512
```

### Compare models

```bash
python compare.py --ar-ckpt checkpoints/ar_best.pt --mdlm-ckpt checkpoints/mdlm_best.pt
```

### Generate text

```bash
python generate.py --checkpoint checkpoints/ar_best.pt --prompt "Once upon a time"
```

## Trained Checkpoints

Trained checkpoints (PyTorch, H100) are available on [Hugging Face](https://huggingface.co/caiovicentino/arche) or via the releases page.

- `AR_best.pt` — AR Transformer (123.6M params, best val loss 1.589 at step 14K)
- `MDLM_best.pt` — Masked Diffusion LM (162.7M params, best val loss 3.412 at step 20K)

## Citation

```bibtex
@article{vicentino2025armdlm,
  title={Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison},
  author={Vicentino, Caio},
  year={2025},
  url={https://github.com/caiovicentino/arche}
}
```

## License

MIT
