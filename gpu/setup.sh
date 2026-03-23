#!/bin/bash
# =============================================================
# Arche — H100 Setup Script
# =============================================================
# Run this on a fresh cloud GPU instance (Lambda, RunPod, etc.)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#   python train_both.py --scale 300M --steps 30000
# =============================================================

set -e

echo "=============================="
echo "  Arche H100 Setup"
echo "=============================="

# Install dependencies
pip install -q torch transformers datasets numpy tqdm wandb

# Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "Setup complete! Run:"
echo ""
echo "  # Quick test (100M, ~30 min):"
echo "  python train_both.py --scale 100M --steps 10000 --batch 32"
echo ""
echo "  # Full experiment (300M, ~2-3 hours):"
echo "  python train_both.py --scale 300M --steps 30000 --batch 32"
echo ""
echo "  # Large scale (700M, ~4-6 hours):"
echo "  python train_both.py --scale 700M --steps 30000 --batch 16"
echo ""
