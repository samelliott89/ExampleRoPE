"""Load checkpoint and generate text."""

import torch
import torch.nn.functional as F
from model import Transformer, Config
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
    print(f"Final training loss: {checkpoint.get('final_loss', 'N/A')}")
    
    return model, config


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 200, temperature: float = 0.8):
    """Generate text from a prompt."""
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_tokens):
        tokens_crop = tokens[:, -model.config.max_seq_len:]
        logits, _ = model(tokens_crop)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    
    return tokenizer.decode(tokens[0].tolist())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="vast_checkpoints/model.pt")
    parser.add_argument("--prompt", type=str, default="It was a dark and stormy night")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    
    # Load model
    model, config = load_checkpoint(args.checkpoint)
    tokenizer = Tokenizer()
    
    # Generate
    print(f"\n--- Prompt: '{args.prompt}' ---\n")
    output = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature)
    print(output)
