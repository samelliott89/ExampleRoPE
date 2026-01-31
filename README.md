# ExampleRoPE

Started to extend previous work by adding influences from more modern techniques (LLaMA/Gemma-style). Began with RoPE but then added GQA, Muon, SwiGLU etc.
Trained on vast.ai with H100 GPUs.

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

## Example
- Able to generate complete sentance structure around themes, but facts are inaccurate and non-sensical. Would require much bigger datasets and longer training/model size.

- **Input**: Once upon a time...
- **Output**: ...in which the state of nature had been at its discretion and by which it was able to take his power into account. The following year, when he came back from England, there were fourteen thousand members who would have believed him but only thirty years later. He became the first man known as the great philosopher and writer of philosophy, whose views on all things are more than facts or ideas. His followers were so far not certain that they could be found out among themselves:

## Files

- `model.py` - Model architecture + training loop
- `inference.py` - Text generation and perplexity evaluation
- `data.py` - FineWeb-Edu/Gutenberg dataset + GPT-2 BPE tokenizer
- `optimizer.py` - Muon + AdamW optimizer



