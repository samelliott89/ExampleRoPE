import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from optimizer import get_optimizer, Muon
from data import load_dataset
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

@dataclass
class Config:
    # Model (roughly 125M params - similar to GPT-2 small)
    max_seq_len: int = 1024
    vocab_size: int = 50257  # GPT-2 BPE vocab size (set from tokenizer)
    dim: int = 768
    hidden_dim: int = 3072  # 4x dim
    n_heads: int = 12
    n_kv_heads: int = 6  # GQA: 2 query heads per KV head (half as many KV heads)
    n_layers: int = 12
    rope_theta: float = 10000.0
    
    # Training (optimized for 92GB VRAM)
    batch_size: int = 64  # reduced - activations use most VRAM, not params
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    epochs: int = 20
    warmup_ratio: float = 0.1  # 10% of training for warmup
    min_lr: float = 6e-5
    grad_clip: float = 1.0
    use_amp: bool = True  # mixed precision (bf16)
    gradient_checkpointing: bool = True  # trade compute for memory
    
    # Logging
    log_interval: int = 10  # log every N batches
    eval_interval: int = 1  # eval every N epochs
    use_wandb: bool = True  # enable wandb logging
    wandb_project: str = "example-rope"  # wandb project name


# --- RoPE ---

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    q_out = q_complex * freqs_cis
    k_out = k_complex * freqs_cis
    q_out = torch.view_as_real(q_out).flatten(-2)
    k_out = torch.view_as_real(k_out).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


# --- Model Components ---

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w3(x)) * self.w1(x))


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_heads % config.n_kv_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, T, n_kv, head_dim = x.shape
        return x[:, :, :, None, :].expand(B, T, n_kv, self.n_rep, head_dim).reshape(B, T, self.n_heads, head_dim)


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config.dim, config.hidden_dim)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.use_checkpoint = config.gradient_checkpointing
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            x = x + torch.utils.checkpoint.checkpoint(
                self._attn_forward, self.attention_norm(x), freqs_cis, use_reentrant=False
            )
            x = x + torch.utils.checkpoint.checkpoint(
                self._ffn_forward, self.ffn_norm(x), use_reentrant=False
            )
        else:
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            x = x + self.feed_forward(self.ffn_norm(x))
        return x
    
    def _attn_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        return self.attention(x, freqs_cis)
    
    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x)


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        head_dim = config.dim // config.n_heads
        freqs_cis = precompute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis)
    
    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = tokens.shape
        x = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:T]
        
        for layer in self.layers:
            x = layer(x, freqs_cis)
        
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss


# --- Learning Rate Scheduler ---

def get_lr(step: int, total_steps: int, warmup_steps: int, config: Config) -> float:
    """Cosine decay with warmup."""
    if step < warmup_steps:
        return config.learning_rate * (step + 1) / warmup_steps
    if step >= total_steps:
        return config.min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# --- Training ---

@torch.no_grad()
def estimate_loss(model, train_ds, val_ds, config: Config, eval_batches: int = 50):
    """Estimate loss on train and val sets."""
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    losses = {}
    for name, ds in [("train", train_ds), ("val", val_ds)]:
        total_loss = 0.0
        for _ in range(eval_batches):
            x, y = ds.get_batch(config.batch_size)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config.use_amp):
                _, loss = model(x, y)
            total_loss += loss.item()
        losses[name] = total_loss / eval_batches
    model.train()
    return losses


def train(config: Config):
    # Load data
    train_ds, val_ds, tokenizer = load_dataset(config.max_seq_len, device)
    config.vocab_size = tokenizer.vocab_size
    
    # Create model
    model = Transformer(config).to(device)
    model = torch.compile(model)  # fused kernels, ~1.5-2x speedup
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Calculate training steps
    batches_per_epoch = train_ds.batches_per_epoch(config.batch_size)
    total_steps = config.epochs * batches_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    print(f"Training: {config.epochs} epochs, {batches_per_epoch} batches/epoch, {total_steps} total steps")
    print(f"Batch size: {config.batch_size}, Tokens/batch: {config.batch_size * config.max_seq_len:,}")
    print(f"Mixed precision: {config.use_amp}, Gradient checkpointing: {config.gradient_checkpointing}")
    
    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            config=asdict(config) | {"n_params": n_params, "device": str(device)},
        )
        wandb.watch(model, log="gradients", log_freq=100)
    
    # Get optimizers (Muon for matrices, AdamW for norms)
    optimizers = get_optimizer(model, config)
    
    # Mixed precision setup (bf16 for Blackwell/Ampere+, fp16 for older)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and amp_dtype == torch.float16)
    
    # Training loop
    global_step = 0
    model.train()
    
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_ds.iter_epoch(config.batch_size)):
            # Update learning rate
            lr = get_lr(global_step, total_steps, warmup_steps, config)
            for opt in optimizers:
                for param_group in opt.param_groups:
                    param_group['lr'] = lr * (0.1 if isinstance(opt, Muon) else 1.0)
            
            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config.use_amp):
                logits, loss = model(x, y)
            
            # Backward pass
            for opt in optimizers:
                opt.zero_grad()
            
            if config.use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizers[0])  # unscale for grad clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                for opt in optimizers:
                    opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            
            # Logging
            if batch_idx % config.log_interval == 0:
                print(f"epoch {epoch+1:2d} | batch {batch_idx:4d}/{batches_per_epoch} | loss {loss.item():.4f} | lr {lr:.2e}")
                if config.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/epoch": epoch + batch_idx / batches_per_epoch,
                        "train/step": global_step,
                    })
        
        # End of epoch
        avg_loss = epoch_loss / n_batches
        print(f"epoch {epoch+1:2d} | avg train loss: {avg_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % config.eval_interval == 0:
            losses = estimate_loss(model, train_ds, val_ds, config)
            print(f"epoch {epoch+1:2d} | eval - train: {losses['train']:.4f}, val: {losses['val']:.4f}")
            if config.use_wandb:
                wandb.log({
                    "eval/train_loss": losses["train"],
                    "eval/val_loss": losses["val"],
                    "eval/epoch": epoch + 1,
                })
    
    # Save final checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "config": config,
        "final_loss": avg_loss,
    }
    torch.save(checkpoint, "checkpoints/model.pt")
    print(f"Saved checkpoint to checkpoints/model.pt")
    
    # Finish wandb run
    if config.use_wandb:
        wandb.finish()
    
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 100, temperature: float = 0.8):
    """Generate text from a prompt."""
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_tokens):
        tokens_crop = tokens[:, -model.config.max_seq_len:]
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits, _ = model(tokens_crop)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    
    return tokenizer.decode(tokens[0].tolist())


if __name__ == "__main__":
    # Load .env file (for WANDB_API_KEY etc)
    from dotenv import load_dotenv
    load_dotenv()
    
    config = Config()
    
    # Train
    model, tokenizer = train(config)
    
    # Generate samples
    print("\n" + "="*60)
    print("GENERATION SAMPLES")
    print("="*60)
    
    prompts = [
        "It was a dark and stormy night",
        "The door opened slowly, and",
        "She looked at him with",
    ]
    
    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        output = generate(model, tokenizer, prompt, max_tokens=300, temperature=0.8)
        print(output)