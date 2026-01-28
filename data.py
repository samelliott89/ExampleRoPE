"""BPE tokenized text dataset with Project Gutenberg classics."""

import os
import urllib.request
import torch
import tiktoken

DATA_DIR = "data"
COMBINED_PATH = "data/gutenberg.txt"

# Curated fiction classics from Project Gutenberg (ID -> Title for logging)
GUTENBERG_BOOKS = {
    1342: "Pride and Prejudice",
    11: "Alice's Adventures in Wonderland",
    84: "Frankenstein",
    1661: "The Adventures of Sherlock Holmes",
    2701: "Moby Dick",
    174: "The Picture of Dorian Gray",
    345: "Dracula",
    1080: "A Modest Proposal",
    98: "A Tale of Two Cities",
    1400: "Great Expectations",
    76: "Adventures of Huckleberry Finn",
    74: "The Adventures of Tom Sawyer",
    35: "The Time Machine",
    36: "The War of the Worlds",
    43: "The Strange Case of Dr Jekyll and Mr Hyde",
    120: "Treasure Island",
    1232: "The Prince",
    5200: "Metamorphosis",
    2591: "Grimms' Fairy Tales",
    1952: "The Yellow Wallpaper",
    514: "Little Women",
    161: "Sense and Sensibility",
    158: "Emma",
    105: "Persuasion",
    1260: "Jane Eyre",
    768: "Wuthering Heights",
    16328: "Beowulf",
    2600: "War and Peace",
    28054: "The Brothers Karamazov",
    2554: "Crime and Punishment",
}


def download_book(book_id: int, title: str) -> str:
    """Download a single book from Project Gutenberg."""
    path = os.path.join(DATA_DIR, f"{book_id}.txt")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # Try multiple mirror URLs
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    for url in urls:
        try:
            print(f"  Downloading: {title}...")
            urllib.request.urlretrieve(url, path)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            continue

    print(f"  Failed to download: {title}")
    return ""


def clean_gutenberg_text(text: str) -> str:
    """Remove Gutenberg header/footer boilerplate."""
    # Find start of actual content
    start_markers = ["*** START OF", "***START OF", "*END*THE SMALL PRINT"]
    end_markers = ["*** END OF", "***END OF", "End of Project Gutenberg"]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Find the next newline after the marker
            newline_idx = text.find("\n", idx)
            if newline_idx != -1:
                start_idx = max(start_idx, newline_idx + 1)

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = min(end_idx, idx)

    return text[start_idx:end_idx].strip()


def download_data():
    """Download all Project Gutenberg books."""
    if os.path.exists(COMBINED_PATH):
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {len(GUTENBERG_BOOKS)} books from Project Gutenberg...")

    all_text = []
    for book_id, title in GUTENBERG_BOOKS.items():
        text = download_book(book_id, title)
        if text:
            cleaned = clean_gutenberg_text(text)
            all_text.append(cleaned)

    # Combine all books with separators
    combined = "\n\n".join(all_text)

    with open(COMBINED_PATH, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"Combined {len(all_text)} books into {COMBINED_PATH}")
    print(f"Total size: {len(combined):,} characters ({len(combined) / 1e6:.1f} MB)")


class Tokenizer:
    """BPE tokenizer using tiktoken (GPT-2 encoding)."""

    def __init__(self):
        # gpt2 has 50257 tokens, cl100k_base (GPT-4) has 100k
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
        # Pre-chunk into non-overlapping sequences for proper epochs
        n_seqs = len(data) // (seq_len + 1)
        self.n_seqs = n_seqs
        # Trim data to fit exactly
        trimmed = data[: n_seqs * (seq_len + 1)]
        self.chunks = trimmed.view(n_seqs, seq_len + 1)  # (n_seqs, seq_len+1)

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
                continue  # skip incomplete final batch
            batch = self.chunks[idx]
            x, y = batch[:, :-1], batch[:, 1:]
            yield x.to(self.device), y.to(self.device)

    def batches_per_epoch(self, batch_size: int) -> int:
        return self.n_seqs // batch_size


def load_dataset(
    seq_len: int, device: str = "cpu"
) -> tuple[TextDataset, TextDataset, Tokenizer]:
    """Load dataset, returning train/val splits and tokenizer."""
    download_data()

    with open(COMBINED_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Dataset: {len(text):,} characters ({len(text) / 1e6:.1f} MB)")

    tokenizer = Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size:,} tokens (GPT-2 BPE)")

    # Tokenize (this is much faster than char-level and produces fewer tokens)
    print("Tokenizing...")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Train/val split (90/10)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    compression = len(text) / len(data)
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    print(f"Compression: {compression:.1f}x (chars per token)")

    train_dataset = TextDataset(train_data, seq_len, device)
    val_dataset = TextDataset(val_data, seq_len, device)

    return train_dataset, val_dataset, tokenizer


if __name__ == "__main__":
    train_ds, val_ds, tokenizer = load_dataset(seq_len=1024, device="cpu")

    print("\nDataset stats:")
    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences: {len(val_ds):,}")
    print(f"  Batches/epoch (bs=256): {train_ds.batches_per_epoch(256)}")

    # Test batch
    x, y = train_ds.get_batch(batch_size=4)
    print(f"\nBatch shapes: x={x.shape}, y={y.shape}")

    # Test epoch iteration
    print("\nTesting epoch iteration...")
    for i, (x, y) in enumerate(train_ds.iter_epoch(batch_size=256)):
        if i == 0:
            print(f"  First batch: {x.shape}")
        if i >= 2:
            print(f"  ... ({train_ds.batches_per_epoch(256)} total batches)")
            break

    # Show sample
    print("\n--- Sample (1024 tokens) ---")
    print(tokenizer.decode(x[0].tolist())[:500] + "...")
