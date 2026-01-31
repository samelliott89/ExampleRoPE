# ExampleRoPE

Minimal transformer implementation with modern techniques (LLaMA/Gemma-style). Trained on vast.ai with H100 GPUs.

## Features

- **RoPE** - Rotary Position Embeddings
- **GQA** - Grouped Query Attention (8 query heads, 4 KV heads)
- **QK-Norm** - Query/Key normalization for stability
- **SwiGLU** - Gated MLP activation
- **RMSNorm** - Pre-normalization
- **Logit Softcap** - Caps logits to prevent instability
- **Muon + AdamW** - Orthogonalized momentum for matrices, AdamW for norms
- Mixed precision (bf16), gradient checkpointing, torch.compile
- KV cache for efficient generation
- Wandb logging

## Model Config (~80M params)

| Param | Value |
|-------|-------|
| dim | 512 |
| hidden_dim | 2048 |
| n_layers | 8 |
| n_heads | 8 |
| n_kv_heads | 4 |
| max_seq_len | 512 |
| logit_softcap | 30.0 |

## Training

Trained on FineWeb-Edu (1M samples, ~1B tokens) for 10 epochs on 2x H100 GPUs via vast.ai.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add WANDB_API_KEY
```

## Run

```bash
# Single GPU
python model.py

# Multi-GPU
torchrun --nproc_per_node=4 model.py
```

Saves checkpoints to `checkpoints/`.

## Inference

```bash
# Generate text
python inference.py --checkpoint checkpoints/model.pt --prompt "Once upon a time"

# Evaluate perplexity
python inference.py --checkpoint checkpoints/model.pt --eval
```

## Files

- `model.py` - Model architecture + training loop
- `inference.py` - Text generation and perplexity evaluation
- `data.py` - FineWeb-Edu/Gutenberg dataset + GPT-2 BPE tokenizer
- `optimizer.py` - Muon + AdamW optimizer
