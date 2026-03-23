"""
MDLM vs Autoregressive — Fair Comparison
==========================================
Compares diffusion and autoregressive models trained on the SAME data.

Metrics:
  1. Perplexity (val set) — language modeling quality
  2. Generation quality — human-readable samples
  3. Generation speed — tokens/second during generation
  4. Diversity — unique n-grams in generated text
  5. Repetition — fraction of repeated n-grams

Usage:
    python compare.py
    python compare.py --mdlm-ckpt checkpoints/mdlm_best.npz --ar-ckpt checkpoints/ar_best.npz
"""

import os
import time
import argparse
import math
from collections import Counter

import numpy as np
import mlx.core as mx
import mlx.nn as nn


def load_mdlm(ckpt_path, vocab_size=50257):
    """Load trained MDLM model."""
    from mdlm import MDLM, MDLMConfig
    config = MDLMConfig(d_model=512, n_layers=9, n_heads=8, head_dim=64,
                        ffn_dim=1024, vocab_size=vocab_size)
    model = MDLM(config)
    weights = mx.load(ckpt_path)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model


def load_autoregressive(ckpt_path, vocab_size=50257):
    """Load trained autoregressive model."""
    from config import ArcheConfig
    from model import ArcheModel
    config = ArcheConfig(
        layer_pattern=["gla", "gla", "gla", "attn", "gla", "gla", "ssm", "gla", "gla"],
        vocab_size=vocab_size,
    )
    model = ArcheModel(config)
    weights = mx.load(ckpt_path)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model, config


def compute_ar_perplexity(model, val_chunks, batch_size=4, max_batches=50):
    """Compute perplexity for autoregressive model."""
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, min(len(val_chunks), max_batches * batch_size), batch_size):
        batch = mx.array(val_chunks[i:i + batch_size])
        if batch.shape[0] < 2:
            continue
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        logits, _ = model(input_ids)
        loss = nn.losses.cross_entropy(logits, targets)
        mx.eval(loss)
        total_loss += mx.sum(loss).item()
        total_tokens += targets.size
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def compute_mdlm_loss(model, val_chunks, batch_size=4, max_batches=50):
    """Compute average masked prediction loss for MDLM."""
    total_loss = 0.0
    count = 0
    for i in range(0, min(len(val_chunks), max_batches * batch_size), batch_size):
        batch = mx.array(val_chunks[i:i + batch_size])
        if batch.shape[0] < 2:
            continue
        loss = model.compute_loss(batch)
        mx.eval(loss)
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def measure_generation_speed(generate_fn, n_runs=5, length=128):
    """Measure generation speed in tokens/second."""
    # Warmup
    generate_fn(length=64)

    times = []
    for _ in range(n_runs):
        start = time.time()
        tokens = generate_fn(length=length)
        mx.eval(tokens)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    tok_per_sec = length / avg_time
    return tok_per_sec, avg_time


def compute_text_metrics(texts):
    """Compute diversity and repetition metrics for generated texts."""
    all_tokens = []
    for text in texts:
        words = text.lower().split()
        all_tokens.extend(words)

    if len(all_tokens) < 10:
        return {"unique_ratio": 0, "rep_2gram": 1.0, "rep_3gram": 1.0, "avg_length": 0}

    # Unique token ratio
    unique_ratio = len(set(all_tokens)) / len(all_tokens)

    # N-gram repetition rates
    def ngram_repetition(tokens, n):
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        repeated = sum(1 for c in counts.values() if c > 1)
        return repeated / len(counts)

    rep_2 = ngram_repetition(all_tokens, 2)
    rep_3 = ngram_repetition(all_tokens, 3)

    # Average text length
    avg_len = np.mean([len(t.split()) for t in texts])

    return {
        "unique_ratio": unique_ratio,
        "rep_2gram": rep_2,
        "rep_3gram": rep_3,
        "avg_length": avg_len,
    }


def generate_samples(generate_fn, tokenizer, n=10, length=128):
    """Generate multiple text samples."""
    texts = []
    for i in range(n):
        tokens = generate_fn(length=length)
        mx.eval(tokens)
        if hasattr(tokens, 'tolist'):
            token_list = tokens.tolist()
        else:
            token_list = list(tokens)
        text = tokenizer.decode(token_list)
        texts.append(text)
    return texts


def print_comparison(results):
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("  MDLM vs AUTOREGRESSIVE — Comparison on Same Data")
    print("=" * 70)

    print(f"\n  {'Metric':<30} {'Autoregressive':>18} {'MDLM':>18}")
    print(f"  {'-'*28:<30} {'-'*16:>18} {'-'*16:>18}")

    for key, ar_val, mdlm_val in results:
        if isinstance(ar_val, float):
            print(f"  {key:<30} {ar_val:>18.4f} {mdlm_val:>18.4f}")
        elif isinstance(ar_val, str):
            print(f"  {key:<30} {ar_val:>18} {mdlm_val:>18}")
        else:
            print(f"  {key:<30} {str(ar_val):>18} {str(mdlm_val):>18}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare MDLM vs Autoregressive")
    parser.add_argument("--mdlm-ckpt", default="checkpoints/mdlm_best.npz")
    parser.add_argument("--ar-ckpt", default="checkpoints/ar_baseline_best.npz")
    parser.add_argument("--data", default="data/training_mix.txt")
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Prepare validation data
    print("Preparing validation data...")
    with open(args.data, "r") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    tokens = np.array(tokens[:len(tokens) // 256 * 256], dtype=np.int32)
    chunks = tokens.reshape(-1, 256)
    val_chunks = chunks[-500:]  # Last 500 chunks as val
    print(f"  Val chunks: {len(val_chunks)}")

    results = []

    # --- Load models ---
    print("\nLoading MDLM...")
    mdlm = load_mdlm(args.mdlm_ckpt, tokenizer.vocab_size)
    from mdlm import count_parameters as mdlm_count
    mdlm_params = mdlm_count(mdlm)

    ar_model = None
    ar_params = 0
    if os.path.exists(args.ar_ckpt):
        print("Loading Autoregressive model...")
        ar_model, ar_config = load_autoregressive(args.ar_ckpt, tokenizer.vocab_size)
        from model import count_parameters as ar_count
        ar_params = ar_count(ar_model)
    else:
        print(f"  WARNING: AR checkpoint not found: {args.ar_ckpt}")
        print("  Run: python train.py --data data/training_mix.txt --steps 5000")
        print("  Showing MDLM results only.\n")

    results.append(("Parameters", f"{ar_params/1e6:.1f}M" if ar_model else "N/A",
                     f"{mdlm_params/1e6:.1f}M"))

    # --- Validation Loss ---
    print("\nComputing validation metrics...")
    mdlm_loss = compute_mdlm_loss(mdlm, val_chunks)
    results.append(("Val loss (masked pred)", "N/A", f"{mdlm_loss:.4f}"))

    if ar_model:
        ar_ppl = compute_ar_perplexity(ar_model, val_chunks)
        results.append(("Val perplexity", f"{ar_ppl:.2f}", "N/A (different metric)"))

    # --- Generation Speed ---
    print("Measuring generation speed...")
    from mdlm import sample as mdlm_sample
    mdlm_gen = lambda length=128: mdlm_sample(mdlm, length=length, steps=100,
                                                temperature=0.8, repetition_penalty=1.3)
    mdlm_speed, mdlm_time = measure_generation_speed(mdlm_gen, n_runs=3, length=128)
    results.append(("Gen speed (tok/s)", "", f"{mdlm_speed:.1f}"))
    results.append(("Gen time (128 tok)", "", f"{mdlm_time:.2f}s"))

    if ar_model:
        from generate import generate as ar_generate_fn
        def ar_gen(length=128):
            tokens = mx.array([[tokenizer.encode("Once")[0]]])
            for _ in range(length - 1):
                logits, _ = ar_model(tokens)
                next_tok = mx.argmax(logits[:, -1, :], axis=-1)
                tokens = mx.concatenate([tokens, next_tok[:, None]], axis=1)
            mx.eval(tokens)
            return tokens[0]
        ar_speed, ar_time = measure_generation_speed(ar_gen, n_runs=3, length=128)
        results[-2] = ("Gen speed (tok/s)", f"{ar_speed:.1f}", f"{mdlm_speed:.1f}")
        results[-1] = ("Gen time (128 tok)", f"{ar_time:.2f}s", f"{mdlm_time:.2f}s")

    # --- Generation Quality ---
    print("Generating samples...")
    mdlm_texts = generate_samples(mdlm_gen, tokenizer, n=args.n_samples, length=128)
    mdlm_metrics = compute_text_metrics(mdlm_texts)

    results.append(("Unique word ratio", "", f"{mdlm_metrics['unique_ratio']:.3f}"))
    results.append(("2-gram repetition", "", f"{mdlm_metrics['rep_2gram']:.3f}"))
    results.append(("3-gram repetition", "", f"{mdlm_metrics['rep_3gram']:.3f}"))

    if ar_model:
        ar_texts = generate_samples(ar_gen, tokenizer, n=args.n_samples, length=128)
        ar_metrics = compute_text_metrics(ar_texts)
        results[-3] = ("Unique word ratio", f"{ar_metrics['unique_ratio']:.3f}",
                        f"{mdlm_metrics['unique_ratio']:.3f}")
        results[-2] = ("2-gram repetition", f"{ar_metrics['rep_2gram']:.3f}",
                        f"{mdlm_metrics['rep_2gram']:.3f}")
        results[-1] = ("3-gram repetition", f"{ar_metrics['rep_3gram']:.3f}",
                        f"{mdlm_metrics['rep_3gram']:.3f}")

    # --- Print Results ---
    print_comparison(results)

    # --- Print Sample Texts ---
    print("\n--- MDLM Samples ---")
    for i, text in enumerate(mdlm_texts[:3]):
        print(f"\n  [{i+1}] {text[:250]}")

    if ar_model:
        print("\n--- Autoregressive Samples ---")
        for i, text in enumerate(ar_texts[:3]):
            print(f"\n  [{i+1}] {text[:250]}")

    print()


if __name__ == "__main__":
    main()
