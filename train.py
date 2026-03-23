"""
Arche Training Script
======================
Trains the Arche triple-hybrid model on text data using MLX.

Usage:
    python train.py                              # Train with defaults (TinyShakespeare)
    python train.py --data path/to/text.txt      # Train on custom text file
    python train.py --steps 10000 --batch 8      # Custom training params
    python train.py --resume checkpoints/arche_step_1000.npz  # Resume training

Designed to run efficiently on Apple Silicon (M1/M2/M3/M4) with 16GB+ RAM.
"""

import os
import time
import argparse

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from config import ArcheConfig
from model import ArcheModel, print_model_summary, count_parameters
from data import load_text, prepare_data, infinite_batches, batch_iterator


def loss_fn(model, tokens):
    """
    Combined loss: next-token prediction + multi-token prediction.

    Args:
        model: ArcheModel instance
        tokens: (B, seq_len+1) token IDs

    Returns:
        scalar loss value
    """
    input_ids = tokens[:, :-1]   # (B, seq_len)
    targets = tokens[:, 1:]      # (B, seq_len)

    logits, mtp_logits_list = model(input_ids)

    # Main next-token prediction loss
    main_loss = mx.mean(nn.losses.cross_entropy(logits, targets))

    # Multi-token prediction loss (predict further into the future)
    mtp_loss = mx.array(0.0)
    n_mtp = 0
    for k, mtp_logits in enumerate(mtp_logits_list):
        shift = k + 2  # MTP head 0 predicts token at t+2, head 1 at t+3, etc.
        if shift < tokens.shape[1]:
            tgt = tokens[:, shift:]
            pred = mtp_logits[:, : tgt.shape[1]]
            mtp_loss = mtp_loss + mx.mean(nn.losses.cross_entropy(pred, tgt))
            n_mtp += 1

    if n_mtp > 0:
        mtp_loss = mtp_loss / n_mtp

    total_loss = main_loss + model.config.mtp_weight * mtp_loss
    return total_loss


def evaluate(model, val_chunks, config):
    """Compute average validation loss."""
    total_loss = 0.0
    n_batches = 0

    for batch in batch_iterator(val_chunks, config.batch_size, shuffle=False):
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        logits, _ = model(input_ids)
        loss = mx.mean(nn.losses.cross_entropy(logits, targets))
        mx.eval(loss)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def print_routing_stats(model, val_chunks, config):
    """Print routing/proportion info for MoO or SoftMoO models."""
    from layers import MixtureOfOperations, SoftMixtureOfOperations

    # --- SoftMoO: print learned proportions ---
    props = model.get_learned_proportions()
    if props:
        op_names = SoftMixtureOfOperations.OP_NAMES
        print("  Learned proportions:")
        for li, p in props:
            mx.eval(p)
            pcts = [p[j].item() * 100 for j in range(3)]
            dominant = op_names[int(mx.argmax(p).item())]
            # Visual bar
            bar = ""
            for j in range(3):
                w = int(pcts[j] / 100 * 30)
                bar += ["G", "A", "S"][j] * w
            print(f"    L{li}: [{bar:<30s}] "
                  f"GLA {pcts[0]:5.1f}% | Attn {pcts[1]:5.1f}% | SSM {pcts[2]:5.1f}%  -> {dominant}")
        print()
        return

    # --- MoO: print routing distribution ---
    has_moo = any(isinstance(b.mixer, MixtureOfOperations) for b in model.blocks)
    if not has_moo:
        return

    batch = next(batch_iterator(val_chunks, config.batch_size, shuffle=False))
    input_ids = batch[:, :-1]
    model(input_ids)

    stats = model.get_routing_stats()
    mx.eval(*[c for _, c in stats])

    op_names = MixtureOfOperations.OP_NAMES
    print("  Routing distribution:")
    for layer_idx, counts in stats:
        total = mx.sum(counts).item()
        pcts = [counts[j].item() / total * 100 for j in range(3)]
        ps = [counts[j].item() / total for j in range(3)]
        entropy = -sum(p * np.log(p + 1e-10) for p in ps)
        bar = " | ".join(f"{op_names[j]} {pcts[j]:5.1f}%" for j in range(3))
        print(f"    L{layer_idx}: {bar}  H={entropy:.2f}")
    print()


def cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


def train(config, data_path=None, resume_path=None):
    """Main training loop."""

    # --- Data ---
    text = load_text(data_path)
    train_chunks, val_chunks, tokenizer = prepare_data(text, config.seq_len)

    # --- Model ---
    model = ArcheModel(config)

    if resume_path:
        print(f"Resuming from {resume_path}")
        weights = mx.load(resume_path)
        model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    print_model_summary(model, config)

    # --- Optimizer ---
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Compile the training step for performance
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # --- Training Loop ---
    os.makedirs("checkpoints", exist_ok=True)
    data_iter = infinite_batches(train_chunks, config.batch_size)

    if config.use_soft_moo:
        mode_str = "SoftMoO (learned per-layer proportions)"
        prefix = "soft_moo"
    elif config.use_moo:
        mode_str = "MoO (Mixture of Operations)"
        prefix = "moo"
    else:
        mode_str = f"Fixed ({config.layer_pattern})"
        prefix = "fixed"

    print(f"\nStarting training for {config.max_steps} steps...")
    print(f"  Mode: {mode_str}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Seq length: {config.seq_len}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: Apple Silicon (MLX)\n")

    best_val_loss = float("inf")
    start_time = time.time()
    tokens_seen = 0

    for step in range(1, config.max_steps + 1):
        step_start = time.time()

        # Update learning rate (cosine schedule)
        lr = cosine_lr(step, config.warmup_steps, config.max_steps, config.learning_rate)
        optimizer.learning_rate = mx.array(lr)

        # Forward + backward + update
        batch = next(data_iter)
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)

        # Force computation (MLX is lazy)
        mx.eval(model.parameters(), optimizer.state, loss)

        tokens_seen += config.batch_size * config.seq_len
        step_time = time.time() - step_start

        # --- Logging ---
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = tokens_seen / elapsed
            print(
                f"  step {step:5d}/{config.max_steps} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tok_per_sec:,.0f} | "
                f"{step_time*1000:.0f}ms/step"
            )

        # --- Evaluation ---
        if step % config.eval_interval == 0:
            val_loss = evaluate(model, val_chunks, config)
            print(f"\n  >> Eval at step {step}: val_loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f"checkpoints/arche_{prefix}_best.npz"
                model.save_weights(save_path)
                print(f"  >> New best! Saved to {save_path}")

            # Print routing stats for MoO models
            print_routing_stats(model, val_chunks, config)
            print()

        # --- Checkpointing ---
        if step % config.save_interval == 0:
            save_path = f"checkpoints/arche_{prefix}_step_{step}.npz"
            model.save_weights(save_path)
            model.save_weights(f"checkpoints/arche_{prefix}_latest.npz")
            print(f"  >> Checkpoint saved: {save_path}\n")

    # --- Final save ---
    model.save_weights(f"checkpoints/arche_{prefix}_final.npz")
    model.save_weights(f"checkpoints/arche_{prefix}_latest.npz")

    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Total tokens: {tokens_seen:,}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final checkpoint: checkpoints/arche_final.npz")

    # --- Sample generation ---
    print("\n--- Sample generation ---")
    from generate import generate
    generate(model, tokenizer, "To be, or not to be,", max_tokens=100, temperature=0.8)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train the Arche model")
    parser.add_argument("--data", type=str, default=None, help="Path to training text file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=None, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--n-experts", type=int, default=None, help="Number of MoE experts")
    parser.add_argument("--moo", action="store_true", help="Use Mixture of Operations (learned routing)")
    parser.add_argument("--soft-moo", action="store_true", help="Use SoftMoO (learned per-layer proportions)")

    args = parser.parse_args()

    config = ArcheConfig(use_moo=args.moo, use_soft_moo=args.soft_moo)

    # Override config with command-line args
    if args.steps is not None:
        config.max_steps = args.steps
    if args.batch is not None:
        config.batch_size = args.batch
    if args.seq_len is not None:
        config.seq_len = args.seq_len
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.d_model is not None:
        config.d_model = args.d_model
        config.head_dim = config.d_model // config.n_heads
    if args.n_layers is not None:
        config.n_layers = args.n_layers
        config.layer_pattern = None
        config.__post_init__()
    if args.n_experts is not None:
        config.n_experts = args.n_experts

    train(config, data_path=args.data, resume_path=args.resume)


if __name__ == "__main__":
    main()
