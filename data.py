"""Dataset loader for FineWeb-Edu."""

import os

import torch
import tiktoken

DATA_DIR = "data"
FINEWEB_SAMPLES = 1000000  # ~1B tokens


class Tokenizer:
    """BPE tokenizer using tiktoken (GPT-2 encoding)."""

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab  # 50257

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, indices: list[int]) -> str:
        return self.enc.decode(indices)


class TextDataset:
    """Dataset that provides batches of token sequences with epoch support."""

    def __init__(self, data: torch.Tensor, seq_len: int, device: str = "cpu"):
        self.data = data
        self.seq_len = seq_len
        self.device = device
        # Pre-chunk into non-overlapping sequences
        n_seqs = len(data) // (seq_len + 1)
        self.n_seqs = n_seqs
        trimmed = data[: n_seqs * (seq_len + 1)]
        self.chunks = trimmed.view(n_seqs, seq_len + 1)

    def __len__(self):
        return self.n_seqs

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch (for evaluation)."""
        idx = torch.randint(0, self.n_seqs, (batch_size,))
        batch = self.chunks[idx]
        x, y = batch[:, :-1], batch[:, 1:]
        return x.to(self.device), y.to(self.device)

    def iter_epoch(self, batch_size: int, shuffle: bool = True):
        """Iterate through entire dataset once (one epoch)."""
        indices = torch.randperm(self.n_seqs) if shuffle else torch.arange(self.n_seqs)

        for start in range(0, self.n_seqs, batch_size):
            idx = indices[start : start + batch_size]
            if len(idx) < batch_size:
                continue
            batch = self.chunks[idx]
            x, y = batch[:, :-1], batch[:, 1:]
            yield x.to(self.device), y.to(self.device)

    def batches_per_epoch(self, batch_size: int) -> int:
        return self.n_seqs // batch_size


def download_fineweb_edu(num_samples: int = 100000) -> str:
    """Download FineWeb-Edu sample and return combined text."""
    from datasets import load_dataset as hf_load_dataset

    # Use HF token if available (set HF_TOKEN in .env for faster downloads)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HuggingFace token for faster downloads")

    print(f"Loading FineWeb-Edu ({num_samples:,} samples)...")

    ds = hf_load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        token=hf_token,
    )

    texts = []
    total_chars = 0
    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        texts.append(example["text"])
        total_chars += len(example["text"])
        if i % 10000 == 0 and i > 0:
            print(f"  Loaded {i:,} samples ({total_chars / 1e6:.1f}M chars)")

    print(f"Loaded {len(texts):,} samples ({total_chars / 1e6:.1f}M chars)")
    return "\n\n".join(texts)


def load_dataset(
    seq_len: int, device: str = "cpu", num_samples: int = FINEWEB_SAMPLES
) -> tuple[TextDataset, TextDataset, Tokenizer]:
    """Load FineWeb-Edu dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)

    tokenizer = Tokenizer()

    cache_file = f"data/fineweb_edu_{num_samples}.pt"
    if os.path.exists(cache_file):
        print(f"Loading cached tokenized data from {cache_file}")
        data = torch.load(cache_file)
    else:
        text = download_fineweb_edu(num_samples)
        print(f"Dataset: {len(text):,} characters ({len(text) / 1e6:.1f}M)")

        print("Tokenizing...")
        tokens = tokenizer.encode(text)
        data = torch.tensor(tokens, dtype=torch.long)

        torch.save(data, cache_file)
        print(f"Cached tokenized data to {cache_file}")

    print(f"Vocab size: {tokenizer.vocab_size:,} tokens (GPT-2 BPE)")
    print(f"Total tokens: {len(data):,} ({len(data) / 1e6:.1f}M)")

    # Train/val split (95/5)
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

    return (
        TextDataset(train_data, seq_len, device),
        TextDataset(val_data, seq_len, device),
        tokenizer,
    )


if __name__ == "__main__":
    train_ds, val_ds, tokenizer = load_dataset(seq_len=1024, device="cpu")

    print(f"\nDataset stats:")
    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences: {len(val_ds):,}")
    print(f"  Batches/epoch (bs=64): {train_ds.batches_per_epoch(64)}")

    x, y = train_ds.get_batch(batch_size=4)
    print(f"\nBatch shapes: x={x.shape}, y={y.shape}")

    print(f"\n--- Sample ---")
    print(tokenizer.decode(x[0].tolist())[:500] + "...")
