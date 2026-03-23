"""
Quantitative diversity evaluation: 1000 samples from AR and MDLM.
Metrics: Distinct-1/2/3/4, Self-BLEU, unique word ratio, vocab size.
"""
import sys
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import math, time, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_SAMPLES = 1000
LENGTH = 128
AR_BATCH = 25
MDLM_BATCH = 10

# ── Model definitions (matching checkpoint keys) ────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x):
        T = x.shape[2]
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_e = emb.cos()[None, None, :, :]
        sin_e = emb.sin()[None, None, :, :]
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return x * cos_e + torch.cat([-x2, x1], dim=-1) * sin_e

class SwiGLUFFN(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)
        self.w3 = nn.Linear(d, ff, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class ARBlock(nn.Module):
    def __init__(self, d, nh, hd, ff):
        super().__init__()
        inner = nh * hd
        self.wq = nn.Linear(d, inner, bias=False)
        self.wk = nn.Linear(d, inner, bias=False)
        self.wv = nn.Linear(d, inner, bias=False)
        self.wo = nn.Linear(inner, d, bias=False)
        self.rope = RotaryEmbedding(hd)
        self.ffn = SwiGLUFFN(d, ff)
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.nh, self.hd = nh, hd
    def forward(self, x):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B,T,self.nh,self.hd).transpose(1,2)
        k = self.wk(h).view(B,T,self.nh,self.hd).transpose(1,2)
        v = self.wv(h).view(B,T,self.nh,self.hd).transpose(1,2)
        q, k = self.rope(q), self.rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(o.transpose(1,2).reshape(B,T,-1))
        x = x + self.ffn(self.n2(x))
        return x

class ARModel(nn.Module):
    def __init__(self, V, d=768, nh=12, hd=64, ff=2048, nl=12):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([ARBlock(d, nh, hd, ff) for _ in range(nl)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
    def forward(self, tokens):
        x = self.emb(tokens)
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))

class MDLMBlock(nn.Module):
    def __init__(self, d, nh, hd, ff):
        super().__init__()
        inner = nh * hd
        self.wq = nn.Linear(d, inner, bias=False)
        self.wk = nn.Linear(d, inner, bias=False)
        self.wv = nn.Linear(d, inner, bias=False)
        self.wo = nn.Linear(inner, d, bias=False)
        self.rope = RotaryEmbedding(hd)
        self.ffn = SwiGLUFFN(d, ff)
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.nh, self.hd = nh, hd
    def forward(self, x):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B,T,self.nh,self.hd).transpose(1,2)
        k = self.wk(h).view(B,T,self.nh,self.hd).transpose(1,2)
        v = self.wv(h).view(B,T,self.nh,self.hd).transpose(1,2)
        q, k = self.rope(q), self.rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.wo(o.transpose(1,2).reshape(B,T,-1))
        x = x + self.ffn(self.n2(x))
        return x

class TimestepEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Linear(d, d)
    def forward(self, t):
        half = self.d // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return F.silu(self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1)))

class MDLMModel(nn.Module):
    def __init__(self, V, d=768, nh=12, hd=64, ff=2048, nl=12):
        super().__init__()
        self.V = V
        self.mask_id = V
        self.emb = nn.Embedding(V + 1, d)
        self.t_emb = TimestepEmb(d)
        self.blocks = nn.ModuleList([MDLMBlock(d, nh, hd, ff) for _ in range(nl)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
    def forward(self, tokens, t):
        x = self.emb(tokens) + self.t_emb(t)[:, None, :]
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))


# ── Batched generation ──────────────────────────────────────────────────

@torch.no_grad()
def generate_ar_batch(model, tokenizer, batch_size, length, device):
    """Generate a batch of AR samples."""
    start_tok = tokenizer.encode("Once")[0]
    x = torch.full((batch_size, 1), start_tok, dtype=torch.long, device=device)
    for _ in range(length - 1):
        logits = model(x)
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        x = torch.cat([x, next_tok], dim=1)
    return [tokenizer.decode(x[i].tolist()) for i in range(batch_size)]


@torch.no_grad()
def generate_mdlm_single(model, length, steps, device):
    """Generate one MDLM sample."""
    x = torch.full((1, length), model.mask_id, dtype=torch.long, device=device)
    is_masked = torch.ones(length, dtype=torch.bool, device=device)
    token_counts = {}
    for step in range(steps):
        n_masked = is_masked.sum().item()
        if n_masked == 0: break
        t_val = 1.0 - step / steps
        temp = 0.8 * (0.4 + 0.6 * t_val)
        t = torch.tensor([t_val], device=device)
        logits = model(x, t)[0]
        if token_counts:
            for tok_id, count in token_counts.items():
                logits[:, tok_id] /= min(1.3 ** count, 5.0)
        logits /= max(temp, 0.1)
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        confidence = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
        confidence[~is_masked] = 0.0
        steps_left = steps - step
        n_unmask = max(1, n_masked // max(steps_left, 1))
        _, top_idx = confidence.topk(min(n_unmask, n_masked))
        for idx in top_idx:
            i = idx.item()
            tok = sampled[i].item()
            x[0, i] = tok
            is_masked[i] = False
            token_counts[tok] = token_counts.get(tok, 0) + 1
    if is_masked.any():
        logits = model(x, torch.tensor([0.0], device=device))[0]
        x[0, is_masked] = logits[is_masked].argmax(dim=-1)
    return x[0]


# ── Metrics ─────────────────────────────────────────────────────────────

def tokenize_words(texts):
    """Tokenize texts into word lists."""
    return [text.lower().split() for text in texts]

def distinct_n(word_lists, n):
    """Distinct-n: ratio of unique n-grams to total n-grams across all samples."""
    all_ngrams = []
    for words in word_lists:
        for i in range(len(words) - n + 1):
            all_ngrams.append(tuple(words[i:i+n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

def self_bleu(word_lists, n_refs=50):
    """Self-BLEU: average BLEU of each sample against random others. Lower = more diverse."""
    from collections import Counter

    def simple_bleu(candidate, references, max_n=4):
        """Simple corpus-level BLEU without smoothing."""
        scores = []
        for n in range(1, max_n + 1):
            cand_ngrams = Counter()
            for i in range(len(candidate) - n + 1):
                cand_ngrams[tuple(candidate[i:i+n])] += 1

            # Max counts from references
            max_ref = Counter()
            for ref in references:
                ref_ngrams = Counter()
                for i in range(len(ref) - n + 1):
                    ref_ngrams[tuple(ref[i:i+n])] += 1
                for ng, c in ref_ngrams.items():
                    max_ref[ng] = max(max_ref[ng], c)

            clipped = sum(min(c, max_ref.get(ng, 0)) for ng, c in cand_ngrams.items())
            total = sum(cand_ngrams.values())
            if total == 0:
                scores.append(0.0)
            else:
                scores.append(clipped / total)

        if any(s == 0 for s in scores):
            return 0.0
        log_avg = sum(math.log(s) for s in scores) / len(scores)

        # Brevity penalty (against average ref length)
        avg_ref_len = sum(len(r) for r in references) / max(len(references), 1)
        bp = min(1.0, math.exp(1 - avg_ref_len / max(len(candidate), 1)))

        return bp * math.exp(log_avg)

    n = len(word_lists)
    if n < 2:
        return 0.0

    bleu_scores = []
    indices = list(range(n))

    # Sample 200 candidates for efficiency
    sample_indices = random.sample(indices, min(200, n))

    for i in sample_indices:
        # Pick random references (excluding self)
        others = [j for j in indices if j != i]
        refs = random.sample(others, min(n_refs, len(others)))
        ref_texts = [word_lists[j] for j in refs]
        score = simple_bleu(word_lists[i], ref_texts)
        bleu_scores.append(score)

    return sum(bleu_scores) / len(bleu_scores)

def compute_all_metrics(texts):
    """Compute all diversity metrics."""
    word_lists = tokenize_words(texts)

    all_words = [w for wl in word_lists for w in wl]
    unique_words = set(all_words)

    # First-token diversity
    first_words = [wl[0] if wl else "" for wl in word_lists]
    first_5 = [" ".join(wl[:5]) if len(wl) >= 5 else "" for wl in word_lists]

    metrics = {
        "n_samples": len(texts),
        "avg_length_words": sum(len(wl) for wl in word_lists) / len(word_lists),
        "vocab_size": len(unique_words),
        "unique_word_ratio": len(unique_words) / max(len(all_words), 1),
        "distinct_1": distinct_n(word_lists, 1),
        "distinct_2": distinct_n(word_lists, 2),
        "distinct_3": distinct_n(word_lists, 3),
        "distinct_4": distinct_n(word_lists, 4),
        "unique_first_words": len(set(first_words)) / max(len(first_words), 1),
        "unique_first_5grams": len(set(first_5)) / max(len(first_5), 1),
    }

    print("    Computing Self-BLEU (this takes a moment)...")
    metrics["self_bleu"] = self_bleu(word_lists)

    return metrics


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Generating {N_SAMPLES} samples from each model (length={LENGTH})")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    V = 50257

    # Load AR
    print("\nLoading AR...")
    ar = ARModel(V)
    ar.load_state_dict(torch.load("/Users/caiovicentino/Downloads/AR_best.pt",
                                   map_location=DEVICE, weights_only=True))
    ar = ar.to(DEVICE).eval()
    print(f"  OK ({sum(p.numel() for p in ar.parameters())/1e6:.1f}M params)")

    # Generate AR samples
    print(f"\nGenerating {N_SAMPLES} AR samples (batch={AR_BATCH})...")
    ar_texts = []
    t0 = time.time()
    for batch_i in range(N_SAMPLES // AR_BATCH):
        texts = generate_ar_batch(ar, tokenizer, AR_BATCH, LENGTH, DEVICE)
        ar_texts.extend(texts)
        if (batch_i + 1) % 10 == 0:
            elapsed = time.time() - t0
            done = len(ar_texts)
            eta = elapsed / done * (N_SAMPLES - done)
            print(f"  {done}/{N_SAMPLES} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    ar_time = time.time() - t0
    print(f"  Done: {len(ar_texts)} samples in {ar_time:.1f}s")

    # Free AR memory
    del ar
    torch.mps.empty_cache() if DEVICE == "mps" else None

    # Load MDLM
    print("\nLoading MDLM...")
    mdlm = MDLMModel(V)
    mdlm.load_state_dict(torch.load("/Users/caiovicentino/Downloads/MDLM_best.pt",
                                      map_location=DEVICE, weights_only=True))
    mdlm = mdlm.to(DEVICE).eval()
    print(f"  OK ({sum(p.numel() for p in mdlm.parameters())/1e6:.1f}M params)")

    # Generate MDLM samples
    print(f"\nGenerating {N_SAMPLES} MDLM samples...")
    mdlm_texts = []
    t0 = time.time()
    for i in range(N_SAMPLES):
        tokens = generate_mdlm_single(mdlm, LENGTH, 100, DEVICE)
        text = tokenizer.decode(tokens.tolist())
        mdlm_texts.append(text)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (N_SAMPLES - i - 1)
            print(f"  {i+1}/{N_SAMPLES} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    mdlm_time = time.time() - t0
    print(f"  Done: {len(mdlm_texts)} samples in {mdlm_time:.1f}s")

    # Compute metrics
    print("\n" + "="*70)
    print("  COMPUTING METRICS")
    print("="*70)

    print("\n  AR metrics:")
    ar_metrics = compute_all_metrics(ar_texts)

    print("\n  MDLM metrics:")
    mdlm_metrics = compute_all_metrics(mdlm_texts)

    # Print comparison
    print("\n" + "="*70)
    print(f"  DIVERSITY METRICS — {N_SAMPLES} samples, {LENGTH} tokens each")
    print("="*70)
    print(f"\n  {'Metric':<25} {'AR':>12} {'MDLM':>12} {'Winner':>10}")
    print(f"  {'-'*23:<25} {'-'*10:>12} {'-'*10:>12} {'-'*8:>10}")

    for key in ["distinct_1", "distinct_2", "distinct_3", "distinct_4",
                 "self_bleu", "unique_word_ratio", "vocab_size",
                 "unique_first_words", "unique_first_5grams", "avg_length_words"]:
        a = ar_metrics[key]
        m = mdlm_metrics[key]
        # For self_bleu, lower is more diverse; for everything else, higher is more diverse
        if key == "self_bleu":
            winner = "MDLM" if m < a else "AR"
        elif key == "avg_length_words":
            winner = "-"
        else:
            winner = "MDLM" if m > a else "AR"

        if isinstance(a, float):
            print(f"  {key:<25} {a:>12.4f} {m:>12.4f} {winner:>10}")
        else:
            print(f"  {key:<25} {a:>12} {m:>12} {winner:>10}")

    print(f"\n  Generation time:        {ar_time:>10.1f}s {mdlm_time:>10.1f}s")

    # Print sample openings
    print(f"\n  --- AR first 5 openings ---")
    for i in range(5):
        print(f"  [{i+1}] {ar_texts[i][:120]}")
    print(f"\n  --- MDLM first 5 openings ---")
    for i in range(5):
        print(f"  [{i+1}] {mdlm_texts[i][:120]}")

    # Save results
    results = {
        "config": {"n_samples": N_SAMPLES, "length": LENGTH, "device": DEVICE},
        "ar": {"metrics": ar_metrics, "gen_time_s": ar_time,
               "samples_first_10": ar_texts[:10]},
        "mdlm": {"metrics": mdlm_metrics, "gen_time_s": mdlm_time,
                  "samples_first_10": mdlm_texts[:10]},
    }
    with open("paper/diversity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to paper/diversity_results.json")
    print("="*70)


if __name__ == "__main__":
    main()
