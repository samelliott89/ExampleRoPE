import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from optimizer import get_optimizer, Muon

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

@dataclass
class Config:
    # Model
    max_seq_len: int = 256
    vocab_size: int = 10000
    dim: int = 128
    hidden_dim: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    n_layers: int = 4
    rope_theta: float = 10000.0
    
    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_iters: int = 1000
    warmup_iters: int = 100
    min_lr: float = 3e-5
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100


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
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


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

def get_lr(it: int, config: Config) -> float:
    """Cosine decay with warmup."""
    # Warmup
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / config.warmup_iters
    # After max_iters, return min_lr
    if it >= config.max_iters:
        return config.min_lr
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# --- Data ---

def get_batch(config: Config):
    """Generate a random batch for testing. Replace with real data."""
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len + 1)).to(device)
    x = tokens[:, :-1]  # inputs
    y = tokens[:, 1:]   # targets (shifted by 1)
    return x, y


# --- Training ---

def train(config: Config):
    model = Transformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get optimizers (Muon for matrices, AdamW for norms)
    optimizers = get_optimizer(model, config)
    
    model.train()
    for iter in range(config.max_iters):
        # Update learning rate
        lr = get_lr(iter, config)
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = lr * (0.1 if isinstance(opt, Muon) else 1.0)
        
        # Get batch
        x, y = get_batch(config)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Update weights
        for opt in optimizers:
            opt.step()
        
        # Logging
        if iter % config.log_interval == 0:
            print(f"iter {iter:4d} | loss {loss.item():.4f} | lr {lr:.2e}")
    
    return model


if __name__ == "__main__":
    config = Config()

    # Create model
    transformer = Transformer(config)
    model = transformer
    model = torch.compile(model, backend="aot_eager") # swap between aot_eager and max-autotune, reduce-overhead
    model.to(device)
    
    # Test forward pass
    tokens = torch.randint(0, config.vocab_size, (2, 64)).to(device)
    logits, loss = model(tokens)
    
    print(f"Input:  {tokens.shape}")
    print(f"Output: {logits.shape}")
    print(f"Loss: {loss}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")