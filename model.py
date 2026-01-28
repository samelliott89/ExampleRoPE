import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@dataclass
class Config:
    max_seq_len: int = 1024
    vocab_size: int = 10000
    dim: int = 64
    hidden_dim: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    rope_theta: float = 10000.0

# --- RoPE functions (standalone, not inside class) ---

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute rotation frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)  # (seq_len, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to Q and K."""
    # q, k: (B, T, n_heads, head_dim)
    # freqs_cis: (T, head_dim/2)
    
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim/2)
    
    q_out = q_complex * freqs_cis
    k_out = k_complex * freqs_cis
    
    q_out = torch.view_as_real(q_out).flatten(-2)
    k_out = torch.view_as_real(k_out).flatten(-2)
    
    return q_out.type_as(q), k_out.type_as(k)


# --- Modules ---

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x * rsqrt(mean(x²) + eps) * weight
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config.dim, config.hidden_dim)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual connections
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class FeedForward(nn.Module):
    """SwiGLU MLP block."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w3(x)) * self.w1(x))


class Attention(nn.Module):
    """Grouped Query Attention with RoPE."""
    
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
        
        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE to Q and K (before repeat_kv)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # Expand KV heads to match Q heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        # Transpose for attention
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        
        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, T, n_kv, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(B, T, n_kv, self.n_rep, head_dim)
            .reshape(B, T, self.n_heads, head_dim)
        )

class Transformer(nn.Module):
    """Full transformer model."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Token embeddings (no position embeddings — RoPE handles that)
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final norm and output projection
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Precompute RoPE frequencies
        head_dim = config.dim // config.n_heads
        freqs_cis = precompute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        
        # Embed tokens
        x = self.tok_embeddings(tokens)  # (B, T, dim)
        
        # Get frequencies for this sequence length
        freqs_cis = self.freqs_cis[:T]
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, freqs_cis)
        
        # Final norm and project to vocab
        x = self.norm(x)
        logits = self.output(x)  # (B, T, vocab_size)
        
        return logits


if __name__ == "__main__":
    config = Config()

    # Create model
    transformer = Transformer(config)
    model = transformer
    model.to(device)
    
    # Test forward pass
    tokens = torch.randint(0, config.vocab_size, (2, 64)).to(device)
    logits = model(tokens)
    
    print(f"Input:  {tokens.shape}")
    print(f"Output: {logits.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")