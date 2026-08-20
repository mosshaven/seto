"""Seto data — uint16 binary shards for fast training."""

import os
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


class ShardDataset(Dataset):
    """Memory-mapped uint16 binary shards."""

    def __init__(self, data_dir: str, seq_len: int = 1024, split: str = "train"):
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)

        shard_files = sorted(self.data_dir.glob(f"{split}_*.bin"))
        if not shard_files:
            shard_files = sorted(self.data_dir.glob("*.bin"))

        self.data = np.concatenate([
            np.fromfile(str(f), dtype=np.uint16) for f in shard_files
        ])

        self.n_tokens = len(self.data)
        self.n_sequences = self.n_tokens // (seq_len + 1)

        print(f"Loaded {self.n_tokens:,} tokens from {len(shard_files)} shards")
        print(f"  {self.n_sequences:,} sequences of length {seq_len}")

    def __len__(self) -> int:
        return max(1, self.n_sequences - 1)

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.data[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = np.pad(chunk, (0, self.seq_len + 1 - len(chunk)), constant_values=0)

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}


class StreamingShardDataset(IterableDataset):
    """Streaming from uint16 shards — for large datasets."""

    def __init__(self, data_dir: str, seq_len: int = 1024):
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)

    def __iter__(self):
        shard_files = sorted(self.data_dir.glob("*.bin"))
        for shard_file in shard_files:
            data = np.fromfile(str(shard_file), dtype=np.uint16)
            n_seqs = len(data) // (self.seq_len + 1)

            for i in range(n_seqs):
                start = i * (self.seq_len + 1)
                chunk = data[start:start + self.seq_len + 1]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}


def pack_tokens_to_shards(
    token_ids: List[int],
    output_dir: str,
    shard_size: int = 100_000_000,  # 100M tokens per shard
    split: str = "train",
):
    """Pack token IDs into uint16 binary shards."""
    os.makedirs(output_dir, exist_ok=True)

    tokens = np.array(token_ids, dtype=np.uint16)
    n_shards = (len(tokens) + shard_size - 1) // shard_size

    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, len(tokens))
        shard = tokens[start:end]

        shard_path = os.path.join(output_dir, f"{split}_{i:04d}.bin")
        shard.tofile(shard_path)
        print(f"  Shard {i}: {len(shard):,} tokens -> {shard_path}")

    print(f"Packed {len(tokens):,} tokens into {n_shards} shards")


def pack_from_text_files(
    text_files: List[str],
    tokenizer,
    output_dir: str,
    shard_size: int = 100_000_000,
    split: str = "train",
):
    """Read text files, tokenize, and pack into shards."""
    all_tokens = []

    for f in text_files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            all_tokens.extend(ids)
        except Exception as e:
            print(f"  Warning: failed to process {f}: {e}")

    pack_tokens_to_shards(all_tokens, output_dir, shard_size, split)


def pack_from_hf_dataset(
    dataset_name: str,
    tokenizer,
    output_dir: str,
    text_key: str = "text",
    max_samples: Optional[int] = None,
    shard_size: int = 100_000_000,
    split: str = "train",
):
    """Load HuggingFace dataset, tokenize, and pack into shards."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)

    all_tokens = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        text = row.get(text_key, "")
        if text:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            all_tokens.extend(ids)

        if i % 10000 == 0 and i > 0:
            print(f"  Tokenized {i:,} samples...")

    pack_tokens_to_shards(all_tokens, output_dir, shard_size, split)


def find_shards(data_dir: str, split: str = "train") -> List[str]:
    """Find all shard files in a directory."""
    data_path = Path(data_dir)
    shards = sorted(data_path.glob(f"{split}_*.bin"))
    if not shards:
        shards = sorted(data_path.glob("*.bin"))
    return [str(s) for s in shards]


def create_dataloader(
    dataset,
    batch_size: int = 8,
    num_workers: int = 2,
    distributed: bool = False,
) -> DataLoader:
    sampler = None
    if distributed and isinstance(dataset, Dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and isinstance(dataset, Dataset)),
        num_workers=num_workers if isinstance(dataset, Dataset) else 0,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )
