import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    max_seq_len: int = 1024 # maximum sequence length
    vocab_size: int = 10000 # vocabulary size
    dim: int = 64 # dimension of the model
    hidden_dim: int = 256 # dimension of the hidden layer
    n_heads: int = 8  # number of query heads
    n_kv_heads: int = 4  # number of KV heads (n_heads // n_kv_heads = 2 query heads per KV head)
    dropout: float = 0.1 # dropout rate

class FeedForward(nn.Module):
    """SwiGLU MLP block.
    
    Uses gated linear units with SiLU activation for improved training dynamics.
    """
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # value (project up)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # output (project down)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # gate (project up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w3(x)) * self.w1(x))


class Attention(nn.Module):
    """Grouped Query Attention (GQA) with repetition factor for KV heads.
    
    Query heads are grouped, with each group sharing the same key/value heads.
    When n_kv_heads == n_heads: standard Multi-Head Attention
    When n_kv_heads == 1: Multi-Query Attention
    """
    
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_heads % config.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads  # computed, not configurable
        self.head_dim = config.dim // config.n_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)      # Q projection (dim → n_heads * head_dim)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)   # K projection (dim → n_kv_heads * head_dim) - smaller
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)   # V projection (dim → n_kv_heads * head_dim) - smaller
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)      # Output projection (n_heads * head_dim → dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim) e.g. (2, 10, 64)
        B, T, _ = x.shape
        
        # Project to Q, K, V and reshape into heads
        # Q has more heads than K,V — this is the "grouped" part of GQA
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)      # (B, T, 8, 8)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)  # (B, T, 4, 8) <- fewer heads
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)  # (B, T, 4, 8) <- fewer heads
        
        # Expand KV heads to match Q heads by repeating each KV head n_rep times
        k = self._repeat_kv(k)  # (B, T, 4, 8) -> (B, T, 8, 8)
        v = self._repeat_kv(v)  # (B, T, 4, 8) -> (B, T, 8, 8)
        
        # Transpose for attention: move heads before sequence
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, 8, T, 8)
        
        # Scaled dot-product attention (uses Flash Attention when available)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, 8, T, 8)
        
        # Reshape back: (B, n_heads, T, head_dim) -> (B, T, dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, 64)
        return self.wo(out)  # (B, T, 64)
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match number of query heads.
        
        Example with n_kv_heads=4, n_heads=8, n_rep=2:
          Input:  (B, T, 4, 8)  — 4 KV heads
          Output: (B, T, 8, 8)  — 8 heads (each KV head repeated 2x)
        """
        if self.n_rep == 1:
            return x
        B, T, n_kv, head_dim = x.shape                      # (B, T, 4, 8)
        return (
            x[:, :, :, None, :]                             # (B, T, 4, 1, 8) insert dim for repeat
            .expand(B, T, n_kv, self.n_rep, head_dim)       # (B, T, 4, 2, 8) broadcast to n_rep
            .reshape(B, T, self.n_heads, head_dim)          # (B, T, 8, 8)    merge n_kv and n_rep
        )


if __name__ == "__main__":
    config = Config()
    token_embedding = nn.Embedding(config.vocab_size, config.dim).to(device)
    position_embedding = nn.Embedding(config.max_seq_len, config.dim).to(device)
    x = token_embedding + position_embedding[position]

    
    x = torch.randn(2, 10, config.dim).to(device)
    
    # Test FeedForward
    ff = FeedForward(config.dim, config.hidden_dim).to(device)
    out = ff(x)
    assert out.shape == x.shape
    print(f"SwiGLU: {x.shape} -> {out.shape}")
    
    # Test Attention (GQA)
    attn = Attention(config).to(device)
    out = attn(x)
    assert out.shape == x.shape
    print(f"GQA:    {x.shape} -> {out.shape}")