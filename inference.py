"""Load checkpoint and generate text."""

import torch
import torch.nn.functional as F
from model import Transformer, Config, KVCache
from data import Tokenizer

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
    print(f"Training loss: {checkpoint.get('loss', checkpoint.get('final_loss', 'N/A'))}")
    if "epoch" in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    return model, config


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
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
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Forward only new token
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, _, caches = model(next_token, caches=caches)
    else:
        # No cache: recompute everything each step
        for _ in range(max_tokens):
            tokens_crop = tokens[:, -model.config.max_seq_len:]
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, _, _ = model(tokens_crop)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    
    return tokenizer.decode(tokens[0].tolist())


if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="vast_checkpoints/model.pt")
    parser.add_argument("--prompt", type=str, default="It was a dark and stormy night")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache")
    args = parser.parse_args()
    
    # Load model
    model, config = load_checkpoint(args.checkpoint)
    tokenizer = Tokenizer()
    
    # Generate
    print(f"\n--- Prompt: '{args.prompt}' ---")
    print(f"--- KV Cache: {'disabled' if args.no_cache else 'enabled'} ---\n")
    
    start = time.time()
    output = generate(
        model, tokenizer, args.prompt, args.max_tokens, args.temperature,
        use_cache=not args.no_cache
    )
    elapsed = time.time() - start
    
    print(output)
    print(f"\n--- Generated {args.max_tokens} tokens in {elapsed:.2f}s ({args.max_tokens / elapsed:.1f} tok/s) ---")
