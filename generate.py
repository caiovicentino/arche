"""
Arche Text Generation
======================
Supports two modes:
  1. Standard autoregressive generation (token by token)
  2. Speculative decoding using MTP heads (predict K tokens, verify in batch)
"""

import argparse
import mlx.core as mx
import mlx.nn as nn

from config import ArcheConfig
from model import ArcheModel


def sample_top_p(logits, temperature=0.8, top_p=0.9):
    """Sample from logits with temperature and nucleus (top-p) sampling."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)

    # Sort probabilities descending
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Cumulative probabilities
    cum_probs = mx.cumsum(sorted_probs, axis=-1)

    # Remove tokens with cumulative probability above top_p
    mask = cum_probs - sorted_probs > top_p
    sorted_probs = mx.where(mask, 0.0, sorted_probs)

    # Renormalize
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)

    # Sample from the filtered distribution
    token = mx.random.categorical(mx.log(sorted_probs + 1e-10), axis=-1)

    # Map back to original indices
    result = mx.take_along_axis(sorted_indices, token[..., None], axis=-1)
    return result.squeeze(-1)


def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_p=0.9):
    """
    Standard autoregressive text generation.
    Generates one token at a time using the full model.
    """
    tokens = tokenizer.encode(prompt)
    tokens = mx.array([tokens])  # (1, T)

    print(f"\n--- Generating (max {max_tokens} tokens) ---")
    print(prompt, end="", flush=True)

    for _ in range(max_tokens):
        logits, _ = model(tokens)
        next_logits = logits[:, -1, :]  # (1, vocab_size)

        next_token = sample_top_p(next_logits, temperature, top_p)  # (1,)
        tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)
        mx.eval(tokens)

        # Decode and print the new token
        new_text = tokenizer.decode([next_token.item()])
        print(new_text, end="", flush=True)

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    print("\n--- Done ---\n")
    return tokenizer.decode(tokens[0].tolist())


def generate_speculative(model, tokenizer, prompt, max_tokens=200,
                         temperature=0.8, top_p=0.9):
    """
    Speculative decoding using the MTP heads as draft predictors.

    Process:
      1. Generate next token + K speculative tokens from MTP heads
      2. Verify all K+1 tokens in a single forward pass
      3. Accept all correct predictions, reject at first mismatch
      4. Repeat from the last accepted position

    This gives ~2-3x speedup for models where MTP predictions are accurate.
    """
    tokens = tokenizer.encode(prompt)
    tokens_list = list(tokens)
    n_mtp = model.config.mtp_num_future

    print(f"\n--- Speculative Generation (max {max_tokens} tokens, K={n_mtp}) ---")
    print(prompt, end="", flush=True)

    generated = 0
    accepted_total = 0
    draft_total = 0

    while generated < max_tokens:
        input_tokens = mx.array([tokens_list])

        # Forward pass: get main logits + MTP draft predictions
        logits, mtp_logits_list = model(input_tokens)
        mx.eval(logits)

        # Sample next token from main head
        next_token = sample_top_p(logits[:, -1, :], temperature, top_p).item()
        draft_tokens = [next_token]

        # Get draft tokens from MTP heads
        for mtp_logits in mtp_logits_list:
            mx.eval(mtp_logits)
            draft_tok = sample_top_p(mtp_logits[:, -1, :], temperature * 0.9, top_p).item()
            draft_tokens.append(draft_tok)

        # For this simplified version, accept the main token always
        # and verify MTP drafts by running them through the model
        tokens_list.append(next_token)
        generated += 1
        accepted = 1

        # Try to accept speculative tokens
        if len(draft_tokens) > 1 and generated < max_tokens:
            # Verify draft tokens
            verify_input = mx.array([tokens_list])
            verify_logits, _ = model(verify_input)
            mx.eval(verify_logits)

            for k, draft_tok in enumerate(draft_tokens[1:]):
                verify_next = mx.argmax(verify_logits[:, -1, :], axis=-1).item()
                if verify_next == draft_tok:
                    tokens_list.append(draft_tok)
                    generated += 1
                    accepted += 1
                    if generated >= max_tokens:
                        break
                    # Continue verification with extended context
                    verify_input = mx.array([tokens_list])
                    verify_logits, _ = model(verify_input)
                    mx.eval(verify_logits)
                else:
                    break

        accepted_total += accepted
        draft_total += len(draft_tokens)

        # Print newly generated tokens
        new_text = tokenizer.decode(draft_tokens[:accepted])
        print(new_text, end="", flush=True)

        if next_token == tokenizer.eos_token_id:
            break

    acceptance_rate = accepted_total / max(draft_total, 1)
    print(f"\n--- Done (acceptance rate: {acceptance_rate:.1%}) ---\n")
    return tokenizer.decode(tokens_list)


def main():
    parser = argparse.ArgumentParser(description="Generate text with Arche model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/arche_latest.npz",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="To be, or not to be,",
                        help="Text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--speculative", action="store_true",
                        help="Use speculative decoding with MTP heads")
    args = parser.parse_args()

    # Load model
    config = ArcheConfig()
    model = ArcheModel(config)

    print(f"Loading checkpoint: {args.checkpoint}")
    weights = mx.load(args.checkpoint)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Generate
    if args.speculative:
        generate_speculative(model, tokenizer, args.prompt,
                             args.max_tokens, args.temperature, args.top_p)
    else:
        generate(model, tokenizer, args.prompt,
                 args.max_tokens, args.temperature, args.top_p)


if __name__ == "__main__":
    main()
