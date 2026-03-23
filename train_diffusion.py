"""
MDLM Training Script
=====================
Trains the Masked Diffusion Language Model on prepared data.

Usage:
    python train_diffusion.py                                    # Default config
    python train_diffusion.py --data data/training_mix.txt       # Custom data
    python train_diffusion.py --steps 10000 --batch 8            # Custom params
"""

import os
import time
import argparse

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mdlm import MDLM, MDLMConfig, sample, print_mdlm_summary, count_parameters


def prepare_data(text_path, seq_len, tokenizer_name="gpt2"):
    """Tokenize text and create training chunks."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Tokenizing with {tokenizer_name}...")

    with open(text_path, "r") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    tokens = np.array(tokens, dtype=np.int32)
    print(f"Total tokens: {len(tokens):,}")

    # Create fixed-length chunks
    n_chunks = len(tokens) // seq_len
    tokens = tokens[: n_chunks * seq_len]
    chunks = tokens.reshape(n_chunks, seq_len)

    # 95/5 train/val split (more training data since we have plenty)
    n_val = max(1, n_chunks // 20)
    n_train = n_chunks - n_val

    np.random.seed(42)
    perm = np.random.permutation(n_chunks)
    train_chunks = chunks[perm[:n_train]]
    val_chunks = chunks[perm[n_train:]]

    print(f"Train: {n_train:,} chunks | Val: {n_val:,} chunks | Seq len: {seq_len}")
    return train_chunks, val_chunks, tokenizer


def infinite_batches(chunks, batch_size):
    """Infinite iterator over shuffled batches."""
    n = len(chunks)
    indices = np.arange(n)
    while True:
        np.random.shuffle(indices)
        for i in range(0, n - batch_size + 1, batch_size):
            batch_idx = indices[i: i + batch_size]
            yield mx.array(chunks[batch_idx])


def evaluate(model, val_chunks, batch_size, n_batches=20):
    """Compute average validation loss."""
    total_loss = 0.0
    count = 0
    n = len(val_chunks)
    for i in range(0, min(n_batches * batch_size, n), batch_size):
        batch = mx.array(val_chunks[i: i + batch_size])
        if batch.shape[0] < batch_size:
            continue
        loss = model.compute_loss(batch)
        mx.eval(loss)
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def cosine_lr(step, warmup, max_steps, max_lr, min_lr=1e-5):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


def loss_fn(model, tokens):
    """Loss function for value_and_grad."""
    return model.compute_loss(tokens)


def train(args):
    # --- Data ---
    data_path = args.data or "data/training_mix.txt"
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        print("Run: python prepare_training_data.py")
        return

    train_chunks, val_chunks, tokenizer = prepare_data(data_path, args.seq_len)

    # --- Model ---
    config = MDLMConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        head_dim=args.d_model // args.n_heads,
        ffn_dim=args.d_model * 2,
        n_layers=args.n_layers,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
    )

    model = MDLM(config)
    mx.eval(model.parameters())
    print_mdlm_summary(model)

    # --- Optimizer ---
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        weight_decay=0.01,
    )

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    data_iter = infinite_batches(train_chunks, args.batch)

    # --- Training ---
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Training MDLM for {args.steps} steps...")
    print(f"  Batch: {args.batch} | Seq len: {args.seq_len} | LR: {args.lr}")
    print(f"  Data: {data_path}\n")

    best_val = float("inf")
    start = time.time()
    tokens_seen = 0

    for step in range(1, args.steps + 1):
        step_start = time.time()

        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        optimizer.learning_rate = mx.array(lr)

        batch = next(data_iter)
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        tokens_seen += args.batch * args.seq_len
        step_ms = (time.time() - step_start) * 1000

        if step % args.log_interval == 0:
            elapsed = time.time() - start
            tok_s = tokens_seen / elapsed
            print(f"  step {step:5d}/{args.steps} | "
                  f"loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"tok/s {tok_s:,.0f} | {step_ms:.0f}ms")

        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_chunks, args.batch)
            print(f"\n  >> val_loss = {val_loss:.4f} (best: {min(best_val, val_loss):.4f})")

            if val_loss < best_val:
                best_val = val_loss
                model.save_weights("checkpoints/mdlm_best.npz")
                print(f"  >> New best! Saved.")

            # Generate a sample
            print("  >> Sample generation (50 steps):")
            tokens = sample(model, length=64, steps=50, temperature=0.8)
            mx.eval(tokens)
            text = tokenizer.decode(tokens.tolist())
            print(f"     \"{text[:200]}\"")
            print()

    # Final save
    elapsed = time.time() - start
    model.save_weights("checkpoints/mdlm_final.npz")
    print(f"\nTraining complete!")
    print(f"  Time: {elapsed/60:.1f} min | Tokens: {tokens_seen:,}")
    print(f"  Best val loss: {best_val:.4f}")

    # Final generation
    print("\n--- Final samples ---")
    for i in range(3):
        tokens = sample(model, length=128, steps=100, temperature=0.7)
        mx.eval(tokens)
        text = tokenizer.decode(tokens.tolist())
        print(f"\nSample {i+1}:\n{text[:300]}\n")


def main():
    parser = argparse.ArgumentParser(description="Train MDLM")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=9)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
