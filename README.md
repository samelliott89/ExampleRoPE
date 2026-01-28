# ExampleRoPE

Minimal transformer implementation with modern techniques (LLaMA-style).

## Features

- **RoPE** - Rotary Position Embeddings
- **GQA** - Grouped Query Attention (configurable KV heads)
- **SwiGLU** - Gated MLP
- **RMSNorm** - Pre-normalization
- Mixed precision (bf16), gradient checkpointing, torch.compile
- Wandb logging

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add WANDB_API_KEY
```

## Run

```bash
python model.py  # trains on Project Gutenberg, saves to checkpoints/
```

## Config

Edit `Config` in `model.py`:

| Param | Default | Description |
|-------|---------|-------------|
| dim | 768 | Model dimension |
| n_layers | 12 | Transformer blocks |
| n_heads | 12 | Query heads |
| n_kv_heads | 6 | KV heads (GQA) |
| batch_size | 128 | Batch size |
| epochs | 20 | Training epochs |

## Files

- `model.py` - Model + training loop
- `data.py` - Gutenberg dataset + BPE tokenizer
- `optimizer.py` - Muon + AdamW
