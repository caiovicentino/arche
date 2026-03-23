"""
Arche Benchmark — Controlled Comparison
=========================================
Trains multiple architectural variants on the same data and compares.

Baselines:
  1. all-gla     — 9 GLA layers (pure linear attention)
  2. all-attn    — 9 Attention+MLA layers (pure softmax)
  3. balanced    — 3 GLA + 3 Attn + 3 SSM (equal mix)
  4. fixed-711   — 7 GLA + 1 Attn + 1 SSM (our original design)
  5. moo         — Mixture of Operations (learned routing)
  6. moo-random  — MoO with random routing (ablation)

All models use the same: d_model, MoE FFN, optimizer, data, steps.
The ONLY variable is the mixer configuration.

Usage:
    python benchmark.py                    # Run all baselines (2000 steps each)
    python benchmark.py --steps 3000       # More steps for better convergence
    python benchmark.py --skip-existing    # Skip configs that have saved results
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from config import ArcheConfig
from model import ArcheModel, count_parameters
from data import load_text, prepare_data, infinite_batches, batch_iterator
from train import loss_fn, evaluate, cosine_lr


# ── Experiment Definitions ──────────────────────────────────────────────

EXPERIMENTS = {
    "all-gla": {
        "desc": "Pure GLA (9 layers)",
        "layer_pattern": ["gla"] * 9,
        "use_moo": False,
    },
    "all-attn": {
        "desc": "Pure Attention+MLA (9 layers)",
        "layer_pattern": ["attn"] * 9,
        "use_moo": False,
    },
    "balanced-333": {
        "desc": "Balanced 3:3:3 (GLA/Attn/SSM)",
        "layer_pattern": ["gla", "attn", "ssm"] * 3,
        "use_moo": False,
    },
    "fixed-711": {
        "desc": "Fixed 7:1:1 (original design)",
        "layer_pattern": ["gla", "gla", "gla", "attn", "gla", "gla", "ssm", "gla", "gla"],
        "use_moo": False,
    },
    "moo-learned": {
        "desc": "MoO with learned routing",
        "use_moo": True,
        "moo_random_routing": False,
    },
    "moo-random": {
        "desc": "MoO with RANDOM routing (ablation)",
        "use_moo": True,
        "moo_random_routing": True,
    },
    "soft-moo": {
        "desc": "SoftMoO (learned per-layer proportions)",
        "use_soft_moo": True,
    },
    "all-cattn": {
        "desc": "All CompressedAttn (9 layers, r=4)",
        "layer_pattern": ["cattn"] * 9,
        "use_moo": False,
    },
    "hybrid-cattn": {
        "desc": "Hybrid: 5 CAttn + 2 GLA + 1 Attn + 1 SSM",
        "layer_pattern": ["cattn", "cattn", "gla", "cattn", "attn", "cattn", "gla", "ssm", "cattn"],
        "use_moo": False,
    },
}


def make_config(exp_name, exp_def, steps, batch_size, seq_len):
    """Create config for an experiment."""
    kwargs = {
        "n_layers": 9,
        "d_model": 512,
        "n_heads": 8,
        "head_dim": 64,
        "n_experts": 4,
        "ffn_dim": 1024,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "max_steps": steps,
        "learning_rate": 3e-4,
        "warmup_steps": 100,
        "eval_interval": 500,
        "save_interval": 9999999,  # Don't save intermediate checkpoints
        "log_interval": 50,
        "use_moo": exp_def.get("use_moo", False),
        "moo_random_routing": exp_def.get("moo_random_routing", False),
        "use_soft_moo": exp_def.get("use_soft_moo", False),
    }
    if "layer_pattern" in exp_def:
        kwargs["layer_pattern"] = exp_def["layer_pattern"]

    return ArcheConfig(**kwargs)


def train_experiment(exp_name, config, train_chunks, val_chunks):
    """Train one experiment and return results dict."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  {EXPERIMENTS[exp_name]['desc']}")
    print(f"  Pattern: {config.layer_pattern}")
    print(f"{'='*60}")

    model = ArcheModel(config)
    mx.eval(model.parameters())

    n_params = count_parameters(model)
    print(f"  Params: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    data_iter = infinite_batches(train_chunks, config.batch_size)

    best_val_loss = float("inf")
    losses = []
    start = time.time()

    for step in range(1, config.max_steps + 1):
        lr = cosine_lr(step, config.warmup_steps, config.max_steps, config.learning_rate)
        optimizer.learning_rate = mx.array(lr)

        batch = next(data_iter)
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        if step % config.log_interval == 0:
            losses.append(loss.item())
            elapsed = time.time() - start
            tok_s = step * config.batch_size * config.seq_len / elapsed
            print(f"    step {step:5d}/{config.max_steps} | "
                  f"loss {loss.item():.4f} | tok/s {tok_s:,.0f}")

        if step % config.eval_interval == 0 or step == config.max_steps:
            vl = evaluate(model, val_chunks, config)
            if vl < best_val_loss:
                best_val_loss = vl
            print(f"    >> val_loss = {vl:.4f} (best: {best_val_loss:.4f})")

    elapsed = time.time() - start
    tok_per_sec = config.max_steps * config.batch_size * config.seq_len / elapsed

    # Get SoftMoO proportions if applicable
    proportions = None
    if hasattr(model, 'get_learned_proportions'):
        props = model.get_learned_proportions()
        if props:
            proportions = {}
            for li, p in props:
                mx.eval(p)
                proportions[f"L{li}"] = {
                    "gla": round(p[0].item() * 100, 1),
                    "attn": round(p[1].item() * 100, 1),
                    "ssm": round(p[2].item() * 100, 1),
                }
            # Print proportions
            print("    Learned proportions:")
            for k, v in proportions.items():
                dom = max(v, key=v.get)
                print(f"      {k}: GLA {v['gla']:5.1f}% | Attn {v['attn']:5.1f}% | SSM {v['ssm']:5.1f}% -> {dom}")

    result = {
        "name": exp_name,
        "desc": EXPERIMENTS[exp_name]["desc"],
        "pattern": config.layer_pattern,
        "total_params": n_params,
        "best_val_loss": best_val_loss,
        "final_train_loss": losses[-1] if losses else float("inf"),
        "tok_per_sec": tok_per_sec,
        "time_minutes": elapsed / 60,
        "steps": config.max_steps,
        "proportions": proportions,
    }

    print(f"  -> Done: val={best_val_loss:.4f}, "
          f"{elapsed/60:.1f}min, {tok_per_sec:,.0f} tok/s")

    return result


def print_results_table(results):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS — Controlled Comparison")
    print("=" * 80)

    # Sort by val loss
    results = sorted(results, key=lambda r: r["best_val_loss"])

    print(f"\n  {'Model':<16} {'Val Loss':>9} {'Train Loss':>11} "
          f"{'Params':>8} {'tok/s':>8} {'Time':>7}")
    print(f"  {'-'*14:<16} {'-'*9:>9} {'-'*11:>11} "
          f"{'-'*8:>8} {'-'*8:>8} {'-'*7:>7}")

    best_loss = results[0]["best_val_loss"]
    for r in results:
        delta = r["best_val_loss"] - best_loss
        delta_str = f"(+{delta:.2f})" if delta > 0.005 else "(BEST)"
        print(f"  {r['name']:<16} {r['best_val_loss']:>8.4f}  "
              f"{r['final_train_loss']:>10.4f}  "
              f"{r['total_params']/1e6:>6.1f}M  "
              f"{r['tok_per_sec']:>7,.0f}  "
              f"{r['time_minutes']:>5.1f}m  {delta_str}")

    # Analysis
    print(f"\n  --- Analysis ---")
    winner = results[0]
    print(f"  Best model: {winner['name']} (val_loss={winner['best_val_loss']:.4f})")

    # Check if MoO beats all fixed
    moo = next((r for r in results if r["name"] == "moo-learned"), None)
    fixed = [r for r in results if not r["name"].startswith("moo")]
    if moo and fixed:
        best_fixed = min(fixed, key=lambda r: r["best_val_loss"])
        if moo["best_val_loss"] < best_fixed["best_val_loss"]:
            diff = best_fixed["best_val_loss"] - moo["best_val_loss"]
            print(f"  MoO beats best fixed ({best_fixed['name']}) by {diff:.4f}")
        else:
            diff = moo["best_val_loss"] - best_fixed["best_val_loss"]
            print(f"  Best fixed ({best_fixed['name']}) beats MoO by {diff:.4f}")

    # Check if learned routing beats random
    moo_rand = next((r for r in results if r["name"] == "moo-random"), None)
    if moo and moo_rand:
        diff = moo_rand["best_val_loss"] - moo["best_val_loss"]
        if diff > 0:
            print(f"  Learned routing beats random by {diff:.4f} "
                  f"-> routing signal IS meaningful")
        else:
            print(f"  Random routing matches or beats learned "
                  f"-> routing may not matter")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Arche benchmark")
    parser.add_argument("--steps", type=int, default=2000, help="Steps per experiment")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of experiments to run")
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        choices=["shakespeare", "tinystories", "wikitext2"],
                        help="Dataset to train on")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Limit dataset size (chars)")
    args = parser.parse_args()

    # Load data once
    text = load_text(dataset=args.dataset, max_chars=args.max_chars)
    train_chunks, val_chunks, tokenizer = prepare_data(text, args.seq_len)

    # Select experiments
    if args.only:
        exp_names = [e.strip() for e in args.only.split(",")]
    else:
        exp_names = list(EXPERIMENTS.keys())

    print(f"\nRunning {len(exp_names)} experiments, {args.steps} steps each")
    print(f"Experiments: {exp_names}\n")

    results = []
    for exp_name in exp_names:
        if exp_name not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_name}, skipping")
            continue

        config = make_config(exp_name, EXPERIMENTS[exp_name],
                             args.steps, args.batch, args.seq_len)
        result = train_experiment(exp_name, config, train_chunks, val_chunks)
        results.append(result)

        # Save intermediate results
        with open("checkpoints/benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print_results_table(results)

    # Save final results
    with open("checkpoints/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to checkpoints/benchmark_results.json")


if __name__ == "__main__":
    main()
