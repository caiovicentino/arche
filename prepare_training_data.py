"""
Arche Training Data Preparation
=================================
Combines multiple data sources into a high-quality training mix:

1. Synthetic seed data (hand-crafted, textbook quality)
2. TinyStories (diverse narratives, proven for small LMs)
3. Cosmopedia subset (synthetic textbook data from Mixtral)
4. FineWeb-Edu subset (curated educational web content)

Usage:
    python prepare_training_data.py                    # Default: ~20M tokens
    python prepare_training_data.py --target-tokens 50M  # Larger dataset
    python prepare_training_data.py --sources synthetic,tinystories  # Select sources
"""

import os
import json
import argparse
import random

import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
OUTPUT_PATH = os.path.join(DATA_DIR, "training_mix.txt")


def load_synthetic_seed():
    """Load hand-crafted synthetic data (highest quality tier)."""
    texts = []
    for fname in os.listdir(SYNTHETIC_DIR):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(SYNTHETIC_DIR, fname)
            domain = fname.replace(".jsonl", "")
            with open(fpath, "r") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        texts.append(record["text"])
            print(f"  Loaded {domain}: {len(texts)} examples so far")
    print(f"  Total synthetic seed: {len(texts)} examples, ~{sum(len(t) for t in texts)/1e3:.0f}K chars")
    return texts


def load_tinystories(max_chars=None):
    """Load TinyStories dataset."""
    path = os.path.join(DATA_DIR, "tinystories.txt")
    if not os.path.exists(path):
        print("  Downloading TinyStories...")
        from data import load_text
        load_text(dataset="tinystories", max_chars=max_chars or 20_000_000)

    with open(path, "r") as f:
        text = f.read()
    if max_chars:
        text = text[:max_chars]

    # Split into individual stories
    stories = [s.strip() for s in text.split("\n\n") if len(s.strip()) > 100]
    print(f"  TinyStories: {len(stories)} stories, ~{len(text)/1e6:.1f}M chars")
    return stories


def load_cosmopedia(max_examples=5000):
    """Load Cosmopedia subset from HuggingFace (synthetic textbook data)."""
    cache_path = os.path.join(DATA_DIR, "cosmopedia_subset.txt")

    if os.path.exists(cache_path):
        print(f"  Using cached Cosmopedia: {cache_path}")
        with open(cache_path, "r") as f:
            texts = [t.strip() for t in f.read().split("\n===SEPARATOR===\n") if t.strip()]
        return texts

    try:
        from datasets import load_dataset
        print(f"  Downloading Cosmopedia subset ({max_examples} examples)...")
        ds = load_dataset(
            "HuggingFaceTB/cosmopedia",
            "web_samples_v2",
            split="train",
            streaming=True,
        )

        texts = []
        for i, example in enumerate(ds):
            if i >= max_examples:
                break
            text = example.get("text", "").strip()
            if len(text) > 200:
                texts.append(text)

        # Cache
        with open(cache_path, "w") as f:
            f.write("\n===SEPARATOR===\n".join(texts))
        print(f"  Cosmopedia: {len(texts)} examples, ~{sum(len(t) for t in texts)/1e6:.1f}M chars")
        return texts

    except Exception as e:
        print(f"  Cosmopedia download failed: {e}")
        print("  Skipping Cosmopedia (optional source)")
        return []


def load_fineweb_edu(max_examples=5000):
    """Load FineWeb-Edu subset (curated educational web content)."""
    cache_path = os.path.join(DATA_DIR, "fineweb_edu_subset.txt")

    if os.path.exists(cache_path):
        print(f"  Using cached FineWeb-Edu: {cache_path}")
        with open(cache_path, "r") as f:
            texts = [t.strip() for t in f.read().split("\n===SEPARATOR===\n") if t.strip()]
        return texts

    try:
        from datasets import load_dataset
        print(f"  Downloading FineWeb-Edu subset ({max_examples} examples)...")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train",
            streaming=True,
        )

        texts = []
        for i, example in enumerate(ds):
            if i >= max_examples:
                break
            text = example.get("text", "").strip()
            if len(text) > 200:
                texts.append(text)

        with open(cache_path, "w") as f:
            f.write("\n===SEPARATOR===\n".join(texts))
        print(f"  FineWeb-Edu: {len(texts)} examples, ~{sum(len(t) for t in texts)/1e6:.1f}M chars")
        return texts

    except Exception as e:
        print(f"  FineWeb-Edu download failed: {e}")
        print("  Skipping FineWeb-Edu (optional source)")
        return []


def mix_and_prepare(all_sources, target_chars, output_path):
    """
    Mix data sources and write to a single training file.
    Higher quality sources are oversampled (repeated more often).
    """
    # Quality tiers with oversampling weights
    weighted_texts = []
    for text, weight in all_sources:
        for _ in range(weight):
            weighted_texts.append(text)

    random.seed(42)
    random.shuffle(weighted_texts)

    # Write until we reach target size
    total_chars = 0
    with open(output_path, "w") as f:
        idx = 0
        while total_chars < target_chars:
            text = weighted_texts[idx % len(weighted_texts)]
            f.write(text + "\n\n")
            total_chars += len(text) + 2
            idx += 1

    print(f"\nOutput: {output_path}")
    print(f"Total: {total_chars/1e6:.1f}M chars ({total_chars/4:.0f} est. tokens)")
    print(f"Texts used: {idx} (from pool of {len(weighted_texts)})")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data mix")
    parser.add_argument("--target-tokens", type=str, default="20M",
                        help="Target token count (e.g., '10M', '50M')")
    parser.add_argument("--sources", type=str, default="synthetic,tinystories,cosmopedia,fineweb",
                        help="Comma-separated sources to include")
    args = parser.parse_args()

    # Parse target
    target = args.target_tokens.upper().replace("M", "000000").replace("K", "000")
    target_tokens = int(target)
    target_chars = target_tokens * 4  # Rough: ~4 chars per token

    sources_to_use = set(args.sources.split(","))
    print(f"Target: ~{target_tokens/1e6:.0f}M tokens ({target_chars/1e6:.0f}M chars)")
    print(f"Sources: {sources_to_use}\n")

    all_sources = []  # List of (text, weight) tuples

    # 1. Synthetic seed data (highest quality — weight 5x)
    if "synthetic" in sources_to_use:
        print("Loading synthetic seed data...")
        synthetics = load_synthetic_seed()
        for text in synthetics:
            all_sources.append((text, 5))  # 5x oversampling

    # 2. TinyStories (high quality — weight 2x)
    if "tinystories" in sources_to_use:
        print("Loading TinyStories...")
        ts_chars = min(target_chars // 2, 20_000_000)
        stories = load_tinystories(max_chars=ts_chars)
        for text in stories:
            all_sources.append((text, 2))

    # 3. Cosmopedia (good quality — weight 2x)
    if "cosmopedia" in sources_to_use:
        print("Loading Cosmopedia...")
        cosmo = load_cosmopedia(max_examples=3000)
        for text in cosmo:
            all_sources.append((text, 2))

    # 4. FineWeb-Edu (curated web — weight 1x)
    if "fineweb" in sources_to_use:
        print("Loading FineWeb-Edu...")
        fineweb = load_fineweb_edu(max_examples=3000)
        for text in fineweb:
            all_sources.append((text, 1))

    if not all_sources:
        print("ERROR: No data sources loaded!")
        return

    print(f"\nTotal pool: {len(all_sources)} weighted texts")
    mix_and_prepare(all_sources, target_chars, OUTPUT_PATH)
    print("\nDone! Use this file for training:")
    print(f"  python train.py --data {OUTPUT_PATH} --soft-moo --steps 5000")


if __name__ == "__main__":
    main()
