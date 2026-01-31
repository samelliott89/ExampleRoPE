"""Load checkpoint and generate text, or evaluate perplexity."""

import math
import torch
import torch.nn.functional as F
from model import Transformer
from data import Tokenizer, load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = Transformer(config).to(device)

    # Handle torch.compile'd models (keys may have _orig_mod prefix)
    state_dict = checkpoint["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint from {path}")
    print(
        f"Training loss: {checkpoint.get('loss', checkpoint.get('final_loss', 'N/A'))}"
    )
    if "epoch" in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")

    return model, config


def apply_repetition_penalty(
    logits: torch.Tensor, tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Penalize tokens that have already appeared."""
    if penalty == 1.0:
        return logits
    for token_id in set(tokens[0].tolist()):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits


def sample_token(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """Sample a token with temperature, top-k, and top-p (nucleus) filtering."""
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative prob above threshold (keep first token above)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    use_cache: bool = True,
):
    """Generate text from a prompt with optional KV cache."""
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokens = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)

    if use_cache:
        # Initialize KV cache
        caches = model.init_caches(1, device, amp_dtype)

        # Prefill: process entire prompt at once
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits, _, caches = model(tokens, caches=caches)

        # Generate tokens incrementally
        for _ in range(max_tokens):
            next_logits = logits[:, -1, :].clone()
            next_logits = apply_repetition_penalty(
                next_logits, tokens, repetition_penalty
            )
            next_token = sample_token(next_logits, temperature, top_k, top_p)
            tokens = torch.cat([tokens, next_token], dim=1)

            # Forward only new token
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, _, caches = model(next_token, caches=caches)
    else:
        # No cache: recompute everything each step
        for _ in range(max_tokens):
            tokens_crop = tokens[:, -model.config.max_seq_len :]
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, _, _ = model(tokens_crop)
            next_logits = logits[:, -1, :].clone()
            next_logits = apply_repetition_penalty(
                next_logits, tokens, repetition_penalty
            )
            next_token = sample_token(next_logits, temperature, top_k, top_p)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


@torch.no_grad()
def evaluate_perplexity(model, config, n_batches: int = 100):
    """Evaluate perplexity on validation set."""
    _, val_ds, _ = load_dataset(config.max_seq_len, device)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.eval()

    total_loss = 0.0
    batch_size = min(32, config.batch_size)  # smaller batch for eval

    for i, (x, y) in enumerate(val_ds.iter_epoch(batch_size)):
        if i >= n_batches:
            break
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            _, loss, _ = model(x, y)
        total_loss += loss.item()

    avg_loss = total_loss / min(n_batches, i + 1)
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="vast_checkpoints/model.pt")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate perplexity instead of generating"
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=100,
        help="Number of batches for perplexity eval",
    )
    parser.add_argument("--prompt", type=str, default="It was a dark and stormy night")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k sampling (0 to disable)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (1.0 to disable)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (1.0 to disable)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache")
    args = parser.parse_args()

    # Load model
    model, config = load_checkpoint(args.checkpoint)

    if args.eval:
        # Evaluate perplexity
        print(
            f"\nEvaluating perplexity on validation set ({args.eval_batches} batches)..."
        )
        loss, ppl = evaluate_perplexity(model, config, args.eval_batches)
        print(f"Validation loss: {loss:.4f}")
        print(f"Perplexity: {ppl:.2f}")
    else:
        # Generate text
        tokenizer = Tokenizer()

        print(f"\n--- Prompt: '{args.prompt}' ---")
        print(
            f"--- temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, rep_pen={args.repetition_penalty} ---\n"
        )

        start = time.time()
        output = generate(
            model,
            tokenizer,
            args.prompt,
            args.max_tokens,
            args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_cache=not args.no_cache,
        )
        elapsed = time.time() - start

        print(output)
        print(
            f"\n--- Generated {args.max_tokens} tokens in {elapsed:.2f}s ({args.max_tokens / elapsed:.1f} tok/s) ---"
        )
