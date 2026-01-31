"""Dataset loader with multiple dataset options."""

import os
import re
import urllib.request
from enum import Enum
from typing import Optional

import torch
import tiktoken

DATA_DIR = "data"


class DatasetType(Enum):
    GUTENBERG = "gutenberg"
    FINEWEB_EDU = "fineweb_edu"


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
        trimmed = data[:n_seqs * (seq_len + 1)]
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


GUTENBERG_BOOKS = [
    ("84", "Frankenstein"),
    ("1342", "Pride and Prejudice"),
    ("11", "Alice in Wonderland"),
    ("1661", "Sherlock Holmes"),
    ("98", "A Tale of Two Cities"),
    ("2701", "Moby Dick"),
    ("1952", "The Yellow Wallpaper"),
    ("174", "Dorian Gray"),
    ("345", "Dracula"),
    ("1400", "Great Expectations"),
]


def download_book(book_id: str, title: str) -> Optional[str]:
    """Download a book from Project Gutenberg."""
    cache_path = f"{DATA_DIR}/gutenberg_{book_id}.txt"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return f.read()
    
    mirrors = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    ]
    
    for url in mirrors:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8-sig")
                with open(cache_path, "w") as f:
                    f.write(text)
                print(f"  Downloaded: {title}")
                return text
        except Exception:
            continue
    
    print(f"  Failed: {title}")
    return None


def clean_gutenberg_text(text: str) -> str:
    """Remove Gutenberg boilerplate."""
    start_markers = ["*** START OF", "***START OF"]
    end_markers = ["*** END OF", "***END OF", "End of the Project"]
    
    for marker in start_markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            if "\n" in text:
                text = text.split("\n", 1)[1]
            break
    
    for marker in end_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def download_gutenberg() -> str:
    """Download Gutenberg classics and return combined text."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading Project Gutenberg classics...")
    
    texts = []
    for book_id, title in GUTENBERG_BOOKS:
        text = download_book(book_id, title)
        if text:
            texts.append(clean_gutenberg_text(text))
    
    return "\n\n".join(texts)


def load_dataset_gutenberg(
    seq_len: int, 
    device: str = "cpu"
) -> tuple["TextDataset", "TextDataset", "Tokenizer"]:
    """Load Project Gutenberg classics dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tokenizer = Tokenizer()
    
    cache_file = "data/gutenberg_all.pt"
    if os.path.exists(cache_file):
        print(f"Loading cached Gutenberg data from {cache_file}")
        data = torch.load(cache_file)
    else:
        text = download_gutenberg()
        print(f"Gutenberg: {len(text):,} characters ({len(text) / 1e6:.1f}M)")
        
        print("Tokenizing...")
        tokens = tokenizer.encode(text)
        data = torch.tensor(tokens, dtype=torch.long)
        
        torch.save(data, cache_file)
        print(f"Cached to {cache_file}")
    
    print(f"Vocab size: {tokenizer.vocab_size:,} tokens (GPT-2 BPE)")
    print(f"Total tokens: {len(data):,} ({len(data) / 1e6:.1f}M)")
    
    split_idx = int(len(data) * 0.9)
    train_data, val_data = data[:split_idx], data[split_idx:]
    
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    
    return (
        TextDataset(train_data, seq_len, device),
        TextDataset(val_data, seq_len, device),
        tokenizer
    )


def download_fineweb_edu(num_samples: int = 100000) -> str:
    """Download FineWeb-Edu sample and return combined text."""
    from datasets import load_dataset as hf_load_dataset
    
    # Use HF token if available (set HF_TOKEN in .env for faster downloads)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HuggingFace token for faster downloads")
    
    print(f"Loading FineWeb-Edu ({num_samples:,} samples)...")
    
    # Load streaming dataset and take a sample
    ds = hf_load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # 10B token sample
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


def load_dataset_fineweb(
    seq_len: int, 
    device: str = "cpu",
    num_samples: int = 100000
) -> tuple[TextDataset, TextDataset, Tokenizer]:
    """Load FineWeb-Edu dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    tokenizer = Tokenizer()
    
    # Check for cached tokenized data
    cache_file = f"data/fineweb_edu_{num_samples}.pt"
    if os.path.exists(cache_file):
        print(f"Loading cached tokenized data from {cache_file}")
        data = torch.load(cache_file)
    else:
        # Download and tokenize
        text = download_fineweb_edu(num_samples)
        print(f"Dataset: {len(text):,} characters ({len(text) / 1e6:.1f}M)")
        
        print("Tokenizing...")
        tokens = tokenizer.encode(text)
        data = torch.tensor(tokens, dtype=torch.long)
        
        # Cache for next time
        torch.save(data, cache_file)
        print(f"Cached tokenized data to {cache_file}")
    
    print(f"Vocab size: {tokenizer.vocab_size:,} tokens (GPT-2 BPE)")
    print(f"Total tokens: {len(data):,} ({len(data) / 1e6:.1f}M)")
    
    # Train/val split (95/5 for larger dataset)
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    
    train_dataset = TextDataset(train_data, seq_len, device)
    val_dataset = TextDataset(val_data, seq_len, device)
    
    return train_dataset, val_dataset, tokenizer


# Change this to switch datasets
ACTIVE_DATASET = DatasetType.FINEWEB_EDU
FINEWEB_SAMPLES = 1000000  # ~1B tokens


def load_dataset(seq_len: int, device: str = "cpu") -> tuple[TextDataset, TextDataset, Tokenizer]:
    """Load the active dataset."""
    print(f"Dataset: {ACTIVE_DATASET.value}")
    
    if ACTIVE_DATASET == DatasetType.GUTENBERG:
        return load_dataset_gutenberg(seq_len, device)
    elif ACTIVE_DATASET == DatasetType.FINEWEB_EDU:
        return load_dataset_fineweb(seq_len, device, num_samples=FINEWEB_SAMPLES)
    else:
        raise ValueError(f"Unknown dataset: {ACTIVE_DATASET}")


if __name__ == "__main__":
    # Test current active dataset
    train_ds, val_ds, tokenizer = load_dataset(seq_len=1024, device="cpu")
    
    print(f"\nDataset stats:")
    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences: {len(val_ds):,}")
    print(f"  Batches/epoch (bs=64): {train_ds.batches_per_epoch(64)}")
    
    x, y = train_ds.get_batch(batch_size=4)
    print(f"\nBatch shapes: x={x.shape}, y={y.shape}")
    
    print(f"\n--- Sample ---")
    print(tokenizer.decode(x[0].tolist())[:500] + "...")
    
    print(f"\n--- To switch datasets ---")
    print(f"Edit ACTIVE_DATASET in data.py:")
    print(f"  ACTIVE_DATASET = DatasetType.GUTENBERG     # classics")
    print(f"  ACTIVE_DATASET = DatasetType.FINEWEB_EDU   # educational web")
