import math
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from optimizer import get_optimizer, Muon
from data import load_dataset


# --- KV Cache ---


@dataclass
class KVCache:
    """Cache for key-value tensors during autoregressive generation."""
    k: torch.Tensor  # (B, n_kv_heads, cached_len, head_dim)
    v: torch.Tensor  # (B, n_kv_heads, cached_len, head_dim)
    
    @staticmethod
    def empty(
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "KVCache":
        """Create empty cache."""
        return KVCache(
            k=torch.empty(batch_size, n_kv_heads, 0, head_dim, device=device, dtype=dtype),
            v=torch.empty(batch_size, n_kv_heads, 0, head_dim, device=device, dtype=dtype),
        )
    
    def update(
        self, k_new: torch.Tensor, v_new: torch.Tensor, max_len: Optional[int] = None
    ) -> "KVCache":
        """Append new K,V and optionally apply sliding window."""
        k = torch.cat([self.k, k_new], dim=2)
        v = torch.cat([self.v, v_new], dim=2)
        
        # Sliding window: keep only last max_len positions
        if max_len is not None and k.size(2) > max_len:
            k = k[:, :, -max_len:, :]
            v = v[:, :, -max_len:, :]
        
        return KVCache(k=k, v=v)
    
    @property
    def seq_len(self) -> int:
        return self.k.size(2)


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        return rank, world_size, device
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        return 0, 1, device  # single GPU fallback


rank, world_size, device = setup_distributed()
is_main = rank == 0
if is_main:
    print(f"Using device: {device}, World size: {world_size}")


@dataclass
class Config:
    # Model (roughly 125M params - similar to GPT-2 small)
    max_seq_len: int = 512
    vocab_size: int = 50257  # GPT-2 BPE vocab size (set from tokenizer)
    dim: int = 512
    hidden_dim: int = 2048  # 4x dim
    n_heads: int = 8
    n_kv_heads: int = 4  # GQA: 2 query heads per KV head
    n_layers: int = 8
    logit_softcap: float = 30.0  # cap logits to [-30, 30] for stability
    rope_theta: float = 10000.0
    sliding_window: Optional[int] = None  # None = full attention, int = window size

    # Training
    batch_size: int = 256  # large batch for H100s
    learning_rate: float = 1e-3  # higher lr for smaller model
    weight_decay: float = 0.1
    epochs: int = 10  # 10 epochs with 500M tokens
    warmup_ratio: float = 0.05  # 5% warmup (faster ramp)
    min_lr: float = 6e-5
    grad_clip: float = 1.0
    use_amp: bool = True  # mixed precision (bf16)
    gradient_checkpointing: bool = True  # trade compute for memory
    
    # Logging & Checkpoints
    log_interval: int = 10  # log every N batches
    eval_interval: int = 5  # eval every N epochs
    save_interval: int = 5  # save checkpoint every N epochs
    use_wandb: bool = True
    wandb_project: str = "example-rope"
    resume_from: str = ""  # path to checkpoint to resume from (e.g., "checkpoints/model_epoch005.pt")


# --- RoPE ---


def precompute_freqs_cis(
    dim: int, seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings. start_pos offsets for cached generation."""
    seq_len = q.size(1)
    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
    
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
        self.sliding_window = config.sliding_window

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        # QK-Norm: normalize queries and keys for stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass with optional KV cache.
        
        Args:
            x: Input tensor (B, T, dim) - during cached gen, T=1
            freqs_cis: RoPE frequencies (full sequence length)
            cache: Optional KV cache from previous forward passes
            start_pos: Position offset for RoPE (= cache.seq_len if using cache)
        
        Returns:
            output: (B, T, dim)
            new_cache: Updated cache (or None if not using cache)
        """
        B, T, _ = x.shape
        
        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # QK-Norm: normalize queries and keys for stability
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE with position offset
        q, k = apply_rotary_emb(q, k, freqs_cis, start_pos)
        
        # Transpose for attention: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle cache
        new_cache = None
        if cache is not None:
            # Concat with cached K, V (apply sliding window if set)
            new_cache = cache.update(k, v, max_len=self.sliding_window)
            k, v = new_cache.k, new_cache.v
        
        # Expand KV heads for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Compute attention
        # is_causal=True only works when q_len == k_len (no cache)
        # With cache, we need a custom mask or just use is_causal=False since
        # the new token should attend to all cached + current tokens
        if cache is not None:
            # Single token attending to full cached sequence - no causal mask needed
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads for GQA. x: (B, n_kv_heads, T, head_dim)"""
        if self.n_rep == 1:
            return x
        B, n_kv, T, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(B, n_kv, self.n_rep, T, head_dim)
            .reshape(B, self.n_heads, T, head_dim)
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config.dim, config.hidden_dim)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.use_checkpoint = config.gradient_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        # Gradient checkpointing only during training (no cache)
        if self.use_checkpoint and self.training:
            attn_out, _ = torch.utils.checkpoint.checkpoint(
                self._attn_forward,
                self.attention_norm(x),
                freqs_cis,
                None,  # no cache during training
                0,
                use_reentrant=False,
            )
            x = x + attn_out
            x = x + torch.utils.checkpoint.checkpoint(
                self._ffn_forward, self.ffn_norm(x), use_reentrant=False
            )
            return x, None
        else:
            attn_out, new_cache = self.attention(
                self.attention_norm(x), freqs_cis, cache, start_pos
            )
            x = x + attn_out
            x = x + self.feed_forward(self.ffn_norm(x))
            return x, new_cache

    def _attn_forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[KVCache], start_pos: int
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        return self.attention(x, freqs_cis, cache, start_pos)

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x)


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        head_dim = config.dim // config.n_heads
        freqs_cis = precompute_freqs_cis(
            head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        caches: Optional[list[KVCache]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[KVCache]]]:
        """
        Forward pass with optional KV cache for efficient generation.
        
        Args:
            tokens: Input token IDs (B, T)
            targets: Target token IDs for loss computation (B, T)
            caches: List of KVCache per layer (for incremental decoding)
        
        Returns:
            logits: (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided
            new_caches: Updated caches if caches were provided
        """
        B, T = tokens.shape
        x = self.tok_embeddings(tokens)
        
        # Compute start position from cache
        start_pos = 0 if caches is None else caches[0].seq_len
        
        new_caches = [] if caches is not None else None
        
        for i, layer in enumerate(self.layers):
            layer_cache = caches[i] if caches is not None else None
            x, new_cache = layer(x, self.freqs_cis, layer_cache, start_pos)
            if new_caches is not None:
                new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.output(x)
        
        # Logit softcap: prevent extreme values for stability
        if self.config.logit_softcap > 0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_caches
    
    def init_caches(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.bfloat16
    ) -> list[KVCache]:
        """Initialize empty KV caches for all layers."""
        head_dim = self.config.dim // self.config.n_heads
        return [
            KVCache.empty(batch_size, self.config.n_kv_heads, head_dim, device, dtype)
            for _ in range(self.config.n_layers)
        ]


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
                _, loss, _ = model(x, y)  # ignore caches
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
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if config.resume_from and os.path.exists(config.resume_from):
        if is_main:
            print(f"Resuming from {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        state_dict = ckpt["model"]
        # Handle torch.compile'd models (keys may have _orig_mod prefix)
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        start_epoch = ckpt.get("epoch", 0)
        if is_main:
            print(f"Loaded weights from epoch {start_epoch}, loss was {ckpt.get('loss', 'N/A')}")
    
    model = torch.compile(model)  # fused kernels, ~1.5-2x speedup
    
    # Wrap in DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
        print(f"[v4] dim={config.dim}, layers={config.n_layers}, QK-Norm, softcap={config.logit_softcap}")

    # Calculate training steps
    batches_per_epoch = train_ds.batches_per_epoch(config.batch_size)
    total_steps = config.epochs * batches_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)

    if is_main:
        print(f"Training: {config.epochs} epochs, {batches_per_epoch} batches/epoch, {total_steps} total steps")
        print(f"Batch size: {config.batch_size} (x{world_size} GPUs = {config.batch_size * world_size} effective)")
        print(f"Tokens/batch: {config.batch_size * config.max_seq_len * world_size:,}")
        print(f"Mixed precision: {config.use_amp}, Gradient checkpointing: {config.gradient_checkpointing}")

    # Initialize wandb (main process only)
    if config.use_wandb and is_main:
        wandb.init(
            project=config.wandb_project,
            config=asdict(config) | {"n_params": n_params, "device": str(device), "world_size": world_size},
        )
        # wandb.watch(model, log="gradients", log_freq=100)  # disabled - can cause DDP hangs
    
    # Sync all processes before training
    if world_size > 1:
        dist.barrier()

    # Get optimizers (Muon for matrices, AdamW for norms)
    optimizers = get_optimizer(model, config)

    # Mixed precision setup (bf16 for Blackwell/Ampere+, fp16 for older)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.use_amp and amp_dtype == torch.float16
    )

    # Training loop
    global_step = 0
    model.train()

    for epoch in range(start_epoch, config.epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(train_ds.iter_epoch(config.batch_size)):
            # Update learning rate
            lr = get_lr(global_step, total_steps, warmup_steps, config)
            for opt in optimizers:
                for param_group in opt.param_groups:
                    param_group["lr"] = lr * (0.1 if isinstance(opt, Muon) else 1.0)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config.use_amp):
                logits, loss, _ = model(x, y)  # no cache during training

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

            # Logging (main process only)
            if batch_idx % config.log_interval == 0 and is_main:
                print(f"epoch {epoch + 1:2d} | batch {batch_idx:4d}/{batches_per_epoch} | loss {loss.item():.4f} | lr {lr:.2e}")
                if config.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/epoch": epoch + batch_idx / batches_per_epoch,
                        "train/step": global_step,
                    })

        # End of epoch
        avg_loss = epoch_loss / n_batches
        if is_main:
            print(f"epoch {epoch + 1:2d} | avg train loss: {avg_loss:.4f}")

        # Evaluation (main process only)
        if (epoch + 1) % config.eval_interval == 0 and is_main:
            losses = estimate_loss(model, train_ds, val_ds, config)
            print(f"epoch {epoch + 1:2d} | eval - train: {losses['train']:.4f}, val: {losses['val']:.4f}")
            if config.use_wandb:
                wandb.log({
                    "eval/train_loss": losses["train"],
                    "eval/val_loss": losses["val"],
                    "eval/epoch": epoch + 1,
                })
        
        # Periodic checkpoint saving
        if (epoch + 1) % config.save_interval == 0 and is_main:
            os.makedirs("checkpoints", exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            checkpoint = {
                "model": model_to_save.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "loss": avg_loss,
            }
            ckpt_path = f"checkpoints/model_epoch{epoch + 1:03d}.pt"
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Save final checkpoint (main process only)
    if is_main:
        os.makedirs("checkpoints", exist_ok=True)
        # Get underlying model if wrapped in DDP
        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {
            "model": model_to_save.state_dict(),
            "config": config,
            "final_loss": avg_loss,
        }
        torch.save(checkpoint, "checkpoints/model.pt")
        print("Saved checkpoint to checkpoints/model.pt")

    # Finish wandb run
    if config.use_wandb and is_main:
        wandb.finish()
    
    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()

    return model, tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    use_cache: bool = True,
):
    """
    Generate text from a prompt.
    
    Args:
        model: Transformer model (can be DDP-wrapped)
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        use_cache: Use KV cache for faster generation (default True)
    """
    # Unwrap DDP if needed
    unwrapped = model.module if hasattr(model, "module") else model
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    tokens = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)  # (1, prompt_len)
    
    if use_cache:
        # Initialize KV cache
        caches = unwrapped.init_caches(1, device, amp_dtype)
        
        # Prefill: process entire prompt at once
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits, _, caches = model(tokens, caches=caches)
        
        # Generate tokens one at a time with cache
        for _ in range(max_tokens):
            # Sample next token from last position
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Forward only the new token, using cache
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, _, caches = model(next_token, caches=caches)
    else:
        # Original behavior: no cache, recompute everything each step
        for _ in range(max_tokens):
            tokens_crop = tokens[:, -unwrapped.config.max_seq_len:]
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, _, _ = model(tokens_crop)
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

    # Generate samples (main process only)
    if is_main:
        print("\n" + "=" * 60)
        print("GENERATION SAMPLES")
        print("=" * 60)

        prompts = [
            "It was a dark and stormy night",
            "The door opened slowly, and",
            "She looked at him with",
        ]

        for prompt in prompts:
            print(f"\n--- Prompt: '{prompt}' ---")
            output = generate(model, tokenizer, prompt, max_tokens=300, temperature=0.8)
            print(output)
