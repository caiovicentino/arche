"""
Arche Data Pipeline
===================
Handles downloading training text, tokenization, and batch generation.
Supports: TinyShakespeare, TinyStories, WikiText-2/103, or local files.
"""

import os
import numpy as np
import mlx.core as mx


DATASETS = {
    "shakespeare": "TinyShakespeare (~338K tokens, 1 author)",
    "tinystories": "TinyStories (~50M tokens, diverse narratives)",
    "wikitext2": "WikiText-2 (~2.4M tokens, Wikipedia articles)",
}


def load_text(path=None, dataset="shakespeare", max_chars=None):
    """
    Load training text.
    - path: local file
    - dataset: "shakespeare", "tinystories", "wikitext2"
    - max_chars: limit text size (useful for large datasets)
    """
    if path and os.path.exists(path):
        print(f"Loading text from {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return text[:max_chars] if max_chars else text

    if dataset == "tinystories":
        return _load_tinystories(max_chars)
    elif dataset == "wikitext2":
        return _load_wikitext2(max_chars)
    else:
        return _load_shakespeare()


def _load_shakespeare():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "input.txt")

    if os.path.exists(filepath):
        print(f"Using cached data: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    print("Downloading TinyShakespeare...")
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, filepath)
    print(f"Saved ({os.path.getsize(filepath) / 1024:.0f} KB)")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _load_tinystories(max_chars=None):
    """Load TinyStories from HuggingFace. ~470MB download, ~50M tokens."""
    cache_path = os.path.join(os.path.dirname(__file__), "data", "tinystories.txt")

    if os.path.exists(cache_path):
        print(f"Using cached TinyStories: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text[:max_chars] if max_chars else text

    print("Downloading TinyStories from HuggingFace (first time only)...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # Concatenate stories with separator
    texts = []
    total_chars = 0
    limit = max_chars or 50_000_000  # Default: ~50M chars ≈ ~12M tokens
    for example in ds:
        story = example["text"].strip()
        if story:
            texts.append(story)
            total_chars += len(story)
            if total_chars >= limit:
                break

    text = "\n\n".join(texts)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved TinyStories ({len(text) / 1e6:.1f}M chars)")
    return text[:max_chars] if max_chars else text


def _load_wikitext2(max_chars=None):
    """Load WikiText-2 from HuggingFace. ~12MB, ~2.4M tokens."""
    cache_path = os.path.join(os.path.dirname(__file__), "data", "wikitext2.txt")

    if os.path.exists(cache_path):
        print(f"Using cached WikiText-2: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text[:max_chars] if max_chars else text

    print("Downloading WikiText-2 from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(x["text"] for x in ds if x["text"].strip())

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved WikiText-2 ({len(text) / 1e6:.1f}M chars)")
    return text[:max_chars] if max_chars else text


def prepare_data(text, seq_len, tokenizer_name="gpt2"):
    """
    Tokenize text and split into training chunks.

    Returns:
        train_chunks: numpy array of shape (n_train, seq_len+1)
        val_chunks:   numpy array of shape (n_val, seq_len+1)
        tokenizer:    the HuggingFace tokenizer instance
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Tokenizing with {tokenizer_name} (vocab_size={tokenizer.vocab_size})...")

    tokens = tokenizer.encode(text)
    tokens = np.array(tokens, dtype=np.int32)
    print(f"Total tokens: {len(tokens):,}")

    # Split into chunks of (seq_len + 1) for (input, target) pairs
    chunk_size = seq_len + 1
    n_chunks = len(tokens) // chunk_size
    tokens = tokens[: n_chunks * chunk_size]
    chunks = tokens.reshape(n_chunks, chunk_size)

    # 90/10 train/val split
    n_val = max(1, n_chunks // 10)
    n_train = n_chunks - n_val

    np.random.seed(42)
    perm = np.random.permutation(n_chunks)
    train_chunks = chunks[perm[:n_train]]
    val_chunks = chunks[perm[n_train:]]

    print(f"Train chunks: {n_train:,} | Val chunks: {n_val:,} | Chunk size: {chunk_size}")
    return train_chunks, val_chunks, tokenizer


def batch_iterator(chunks, batch_size, shuffle=True):
    """
    Yield batches of token chunks as mx.array.
    Each chunk has shape (seq_len + 1,) — split into input/target in training.
    """
    n = len(chunks)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, n - batch_size + 1, batch_size):
        batch_idx = indices[i : i + batch_size]
        yield mx.array(chunks[batch_idx])


def infinite_batches(chunks, batch_size):
    """Infinite iterator over shuffled batches (for training loop)."""
    while True:
        yield from batch_iterator(chunks, batch_size, shuffle=True)
