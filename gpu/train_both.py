"""
Train AR + MDLM on H100 — Full Paper Experiment
==================================================
Trains both models on the same data with the same compute budget.
Runs comparison and saves all results.

Usage:
    python train_both.py                         # Default: 300M scale
    python train_both.py --scale 300M --steps 30000
    python train_both.py --scale 100M --steps 10000  # Quick test
"""

import os
import time
import json
import argparse
import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from models import create_models


def prepare_data(tokenizer_name="gpt2", max_tokens=50_000_000, seq_len=512):
    """Download and prepare training data."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Download TinyStories
    print("Loading datasets...")
    from datasets import load_dataset

    texts = []
    total_chars = 0

    # TinyStories (main source)
    print("  Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for ex in ds:
        story = ex["text"].strip()
        if story and len(story) > 50:
            texts.append(story)
            total_chars += len(story)
            if total_chars >= max_tokens * 4:  # ~4 chars per token
                break
    print(f"  TinyStories: {len(texts)} stories, {total_chars/1e6:.1f}M chars")

    # Cosmopedia subset
    print("  Loading Cosmopedia subset...")
    try:
        ds2 = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2",
                           split="train", streaming=True)
        count = 0
        for ex in ds2:
            text = ex.get("text", "").strip()
            if text and len(text) > 200:
                texts.append(text)
                total_chars += len(text)
                count += 1
                if count >= 10000:
                    break
        print(f"  Cosmopedia: {count} texts")
    except Exception as e:
        print(f"  Cosmopedia skipped: {e}")

    # Tokenize
    print(f"Tokenizing {len(texts)} texts...")
    all_tokens = []
    for text in texts:
        toks = tokenizer.encode(text)
        all_tokens.extend(toks)
        if len(all_tokens) >= max_tokens:
            break

    all_tokens = all_tokens[:max_tokens]
    tokens = np.array(all_tokens, dtype=np.int64)
    print(f"Total tokens: {len(tokens):,}")

    # Create chunks
    n_chunks = len(tokens) // seq_len
    tokens = tokens[:n_chunks * seq_len].reshape(n_chunks, seq_len)

    # Split
    n_val = max(100, n_chunks // 20)
    n_train = n_chunks - n_val

    np.random.seed(42)
    perm = np.random.permutation(n_chunks)
    train_data = torch.tensor(tokens[perm[:n_train]])
    val_data = torch.tensor(tokens[perm[n_train:]])

    print(f"Train: {n_train:,} chunks | Val: {n_val:,} chunks | Seq: {seq_len}")
    return train_data, val_data, tokenizer


def train_model(model, train_data, val_data, name, steps, batch_size, lr,
                warmup, seq_len, device, log_interval=50, eval_interval=2000):
    """Train one model and return results."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Dataloader
    dataset = TensorDataset(train_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Steps: {steps} | Batch: {batch_size} | LR: {lr}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    best_val = float("inf")
    losses = []
    val_losses = []
    start = time.time()
    step = 0

    while step < steps:
        for (batch,) in loader:
            if step >= steps:
                break
            step += 1

            # Learning rate schedule
            if step < warmup:
                lr_t = lr * step / warmup
            else:
                progress = (step - warmup) / max(steps - warmup, 1)
                lr_t = lr * 0.1 + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_t

            batch = batch.to(device)
            loss = model.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % log_interval == 0:
                elapsed = time.time() - start
                tok_s = step * batch_size * seq_len / elapsed
                losses.append(loss.item())
                print(f"  [{name}] step {step:6d}/{steps} | "
                      f"loss {loss.item():.4f} | lr {lr_t:.2e} | "
                      f"tok/s {tok_s:,.0f}")

            if step % eval_interval == 0 or step == steps:
                val_loss = evaluate_model(model, val_data, batch_size, device)
                val_losses.append((step, val_loss))
                is_best = val_loss < best_val
                if is_best:
                    best_val = val_loss
                    torch.save(model.state_dict(), f"checkpoints/{name}_best.pt")
                print(f"  [{name}] >> val_loss = {val_loss:.4f} "
                      f"{'(NEW BEST)' if is_best else ''}")

                # Generate sample if MDLM
                if hasattr(model, 'generate'):
                    tokens = model.generate(length=128, steps=100, device=device)
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained("gpt2")
                    text = tok.decode(tokens.tolist())
                    print(f"  [{name}] >> Sample: \"{text[:200]}\"")
                print()

    elapsed = time.time() - start
    tok_s = steps * batch_size * seq_len / elapsed

    result = {
        "name": name,
        "best_val_loss": best_val,
        "final_train_loss": losses[-1] if losses else float("inf"),
        "tok_per_sec": tok_s,
        "time_minutes": elapsed / 60,
        "steps": steps,
        "val_history": val_losses,
    }

    torch.save(model.state_dict(), f"checkpoints/{name}_final.pt")
    print(f"  [{name}] Done: val={best_val:.4f}, {elapsed/60:.1f}min, {tok_s:,.0f} tok/s\n")
    return result


@torch.no_grad()
def evaluate_model(model, val_data, batch_size, device, max_batches=50):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    count = 0

    for i in range(0, min(len(val_data), max_batches * batch_size), batch_size):
        batch = val_data[i:i + batch_size].to(device)
        if batch.shape[0] < 2:
            continue
        loss = model.compute_loss(batch)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


def compute_generation_metrics(model, tokenizer, device, n_samples=50, length=256):
    """Compute generation quality metrics."""
    print(f"  Generating {n_samples} samples...")
    texts = []

    if hasattr(model, 'generate'):
        # MDLM
        for i in range(n_samples):
            tokens = model.generate(length=length, steps=100, device=device)
            texts.append(tokenizer.decode(tokens.tolist()))
    else:
        # AR
        for i in range(n_samples):
            x = torch.tensor([[tokenizer.encode("Once")[0]]], device=device)
            for _ in range(length - 1):
                logits = model(x)
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                x = torch.cat([x, next_tok], dim=1)
            texts.append(tokenizer.decode(x[0].tolist()))

    # Metrics
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())

    unique_ratio = len(set(all_words)) / max(len(all_words), 1)

    # N-gram repetition
    def rep_rate(tokens, n):
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        return sum(1 for c in counts.values() if c > 1) / len(counts)

    return {
        "unique_word_ratio": unique_ratio,
        "2gram_rep": rep_rate(all_words, 2),
        "3gram_rep": rep_rate(all_words, 3),
        "samples": texts[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="300M", choices=["100M", "300M", "700M"])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=50_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", default=None, help="Train only 'ar' or 'mdlm'")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    # Data
    train_data, val_data, tokenizer = prepare_data(
        max_tokens=args.max_tokens, seq_len=args.seq_len)

    # Models
    ar_model, mdlm_model, cfg = create_models(tokenizer.vocab_size, args.scale)

    results = {}

    # Train AR
    if args.only is None or args.only == "ar":
        results["ar"] = train_model(
            ar_model, train_data, val_data, "ar",
            steps=args.steps, batch_size=args.batch, lr=args.lr,
            warmup=args.warmup, seq_len=args.seq_len, device=args.device)

    # Train MDLM
    if args.only is None or args.only == "mdlm":
        results["mdlm"] = train_model(
            mdlm_model, train_data, val_data, "mdlm",
            steps=args.steps, batch_size=args.batch, lr=args.lr,
            warmup=args.warmup, seq_len=args.seq_len, device=args.device)

    # Generation metrics
    print("\n" + "=" * 60)
    print("  GENERATION QUALITY COMPARISON")
    print("=" * 60)

    if "ar" in results:
        ar_model.load_state_dict(torch.load("checkpoints/ar_best.pt", weights_only=True))
        ar_model = ar_model.to(args.device).eval()
        ar_gen = compute_generation_metrics(ar_model, tokenizer, args.device)
        results["ar"]["generation"] = ar_gen

    if "mdlm" in results:
        mdlm_model.load_state_dict(torch.load("checkpoints/mdlm_best.pt", weights_only=True))
        mdlm_model = mdlm_model.to(args.device).eval()
        mdlm_gen = compute_generation_metrics(mdlm_model, tokenizer, args.device)
        results["mdlm"]["generation"] = mdlm_gen

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — {args.scale} scale, {args.steps} steps")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'AR':>20} {'MDLM':>20}")
    print(f"  {'-'*23:<25} {'-'*18:>20} {'-'*18:>20}")

    for key in ["best_val_loss", "final_train_loss", "tok_per_sec", "time_minutes"]:
        ar_val = results.get("ar", {}).get(key, "N/A")
        mdlm_val = results.get("mdlm", {}).get(key, "N/A")
        if isinstance(ar_val, float):
            print(f"  {key:<25} {ar_val:>20.4f} {mdlm_val:>20.4f}")
        else:
            print(f"  {key:<25} {str(ar_val):>20} {str(mdlm_val):>20}")

    for key in ["unique_word_ratio", "2gram_rep", "3gram_rep"]:
        ar_val = results.get("ar", {}).get("generation", {}).get(key, "N/A")
        mdlm_val = results.get("mdlm", {}).get("generation", {}).get(key, "N/A")
        if isinstance(ar_val, float):
            print(f"  {key:<25} {ar_val:>20.3f} {mdlm_val:>20.3f}")

    # Print samples
    for name in ["ar", "mdlm"]:
        if name in results and "generation" in results[name]:
            print(f"\n  --- {name.upper()} Samples ---")
            for i, s in enumerate(results[name]["generation"]["samples"][:3]):
                print(f"  [{i+1}] {s[:200]}")

    # Save results
    # Convert non-serializable items
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: vv for kk, vv in v.items()
                           if not callable(vv)}
    with open("checkpoints/comparison_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  Results saved to checkpoints/comparison_results.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
