"""
Arche Routing Analysis
=======================
Analyzes the learned routing patterns in a trained MoO model.

Shows:
  1. Per-layer operation distribution (which ops each layer prefers)
  2. Per-position routing (do early/late positions prefer different ops?)
  3. Comparison with fixed baselines
  4. Specialization metrics (entropy, dominance)

Usage:
    python analyze.py --checkpoint checkpoints/arche_moo_best.npz
    python analyze.py --checkpoint checkpoints/arche_moo_best.npz --detailed
"""

import argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from config import ArcheConfig
from model import ArcheModel
from layers import MixtureOfOperations
from data import load_text, prepare_data, batch_iterator


def collect_routing_data(model, chunks, config, n_batches=10):
    """
    Run multiple batches through the model and collect detailed routing decisions.

    Returns dict with:
      - layer_counts: (n_moo_layers, 3) total token counts per operation
      - position_counts: (n_moo_layers, seq_len, 3) counts by position
      - layer_indices: which block indices are MoO
    """
    moo_layers = [(i, b) for i, b in enumerate(model.blocks)
                  if isinstance(b.mixer, MixtureOfOperations)]

    if not moo_layers:
        print("No MoO layers found in this model.")
        return None

    n_moo = len(moo_layers)
    seq_len = config.seq_len
    layer_counts = np.zeros((n_moo, 3))
    position_counts = np.zeros((n_moo, seq_len, 3))

    for batch_idx, batch in enumerate(batch_iterator(chunks, config.batch_size, shuffle=False)):
        if batch_idx >= n_batches:
            break

        input_ids = batch[:, :-1]
        model(input_ids)
        mx.eval(model.parameters())

        for mi, (li, block) in enumerate(moo_layers):
            indices = block.mixer._last_indices  # (B, T)
            mx.eval(indices)
            idx_np = np.array(indices)

            for op in range(3):
                mask = (idx_np == op)
                layer_counts[mi, op] += mask.sum()
                for t in range(min(seq_len, idx_np.shape[1])):
                    position_counts[mi, t, op] += mask[:, t].sum()

    layer_indices = [li for li, _ in moo_layers]
    return {
        "layer_counts": layer_counts,
        "position_counts": position_counts,
        "layer_indices": layer_indices,
    }


def print_layer_distribution(data):
    """Print per-layer operation distribution as text table + bar chart."""
    op_names = ["GLA", "Attn", "SSM"]
    layer_counts = data["layer_counts"]
    layer_indices = data["layer_indices"]

    print("\n" + "=" * 65)
    print("  PER-LAYER OPERATION DISTRIBUTION")
    print("=" * 65)

    for mi, li in enumerate(layer_indices):
        total = layer_counts[mi].sum()
        pcts = layer_counts[mi] / total * 100
        ps = layer_counts[mi] / total
        entropy = -sum(p * np.log(p + 1e-10) for p in ps)
        max_entropy = np.log(3)
        specialization = 1 - entropy / max_entropy  # 0=uniform, 1=one-hot

        dominant = op_names[np.argmax(pcts)]

        # Text bar chart (50 chars wide)
        bar = ""
        for op in range(3):
            width = int(pcts[op] / 100 * 40)
            chars = {"GLA": "G", "Attn": "A", "SSM": "S"}
            bar += chars[op_names[op]] * width

        print(f"  Layer {li:2d}: [{bar:<40s}] "
              f"spec={specialization:.2f} dom={dominant}")
        print(f"           GLA {pcts[0]:5.1f}% | Attn {pcts[1]:5.1f}% | SSM {pcts[2]:5.1f}%")

    print()


def print_position_analysis(data, config):
    """Analyze whether early/middle/late positions prefer different operations."""
    op_names = ["GLA", "Attn", "SSM"]
    pos_counts = data["position_counts"]  # (n_moo, seq_len, 3)
    layer_indices = data["layer_indices"]
    T = config.seq_len

    # Divide sequence into 4 quarters
    quarters = [
        ("Start  [0-25%]", 0, T // 4),
        ("Early [25-50%]", T // 4, T // 2),
        ("Late  [50-75%]", T // 2, 3 * T // 4),
        ("End  [75-100%]", 3 * T // 4, T),
    ]

    print("=" * 65)
    print("  POSITION-DEPENDENT ROUTING (averaged across layers)")
    print("=" * 65)

    # Average across MoO layers
    avg_pos = pos_counts.sum(axis=0)  # (seq_len, 3)

    for name, start, end in quarters:
        region = avg_pos[start:end].sum(axis=0)
        total = region.sum()
        pcts = region / total * 100
        dominant = op_names[np.argmax(pcts)]
        print(f"  {name}: GLA {pcts[0]:5.1f}% | Attn {pcts[1]:5.1f}% | SSM {pcts[2]:5.1f}%  -> {dominant}")

    print()


def print_specialization_summary(data):
    """Print overall specialization metrics."""
    layer_counts = data["layer_counts"]
    n_layers = len(data["layer_indices"])

    print("=" * 65)
    print("  SPECIALIZATION SUMMARY")
    print("=" * 65)

    # Global distribution
    global_counts = layer_counts.sum(axis=0)
    total = global_counts.sum()
    global_pcts = global_counts / total * 100

    print(f"  Global: GLA {global_pcts[0]:.1f}% | Attn {global_pcts[1]:.1f}% | SSM {global_pcts[2]:.1f}%")

    # Average specialization
    specs = []
    for mi in range(n_layers):
        ps = layer_counts[mi] / layer_counts[mi].sum()
        entropy = -sum(p * np.log(p + 1e-10) for p in ps)
        spec = 1 - entropy / np.log(3)
        specs.append(spec)

    avg_spec = np.mean(specs)
    max_spec = np.max(specs)
    min_spec = np.min(specs)

    print(f"  Avg specialization: {avg_spec:.3f} (0=uniform, 1=single-op)")
    print(f"  Range: [{min_spec:.3f}, {max_spec:.3f}]")

    # Cross-layer diversity: do different layers specialize differently?
    dominant_ops = [np.argmax(layer_counts[mi]) for mi in range(n_layers)]
    n_unique = len(set(dominant_ops))
    op_names = ["GLA", "Attn", "SSM"]
    print(f"  Unique dominant ops across layers: {n_unique}/3")
    print(f"  Dominant pattern: {[op_names[d] for d in dominant_ops]}")

    if n_unique >= 2 and avg_spec > 0.1:
        print("\n  >> FINDING: The model learned to use DIFFERENT operations")
        print("     at different layers. This suggests operation-type routing")
        print("     discovers meaningful computational specialization.")
    elif avg_spec < 0.05:
        print("\n  >> FINDING: Routing is near-uniform (no specialization).")
        print("     The model may need more training or stronger routing signal.")
    else:
        print(f"\n  >> FINDING: Partial specialization detected.")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze Arche MoO routing")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/arche_moo_best.npz",
                        help="Path to trained MoO checkpoint")
    parser.add_argument("--n-batches", type=int, default=20,
                        help="Number of batches to analyze")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed per-position analysis")
    args = parser.parse_args()

    config = ArcheConfig(use_moo=True)
    model = ArcheModel(config)

    print(f"Loading checkpoint: {args.checkpoint}")
    weights = mx.load(args.checkpoint)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    text = load_text()
    _, val_chunks, _ = prepare_data(text, config.seq_len)

    print(f"Analyzing routing on {args.n_batches} batches...")
    data = collect_routing_data(model, val_chunks, config, n_batches=args.n_batches)

    if data is None:
        return

    print_layer_distribution(data)
    if args.detailed:
        print_position_analysis(data, config)
    print_specialization_summary(data)


if __name__ == "__main__":
    main()
