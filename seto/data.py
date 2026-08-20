"""Seto dataset loading for Kaggle and local training."""

import os
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


class PretrainDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 2048, split: str = "train"):
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob(f"*.{split}.*"))

        self.data = []
        for f in self.files:
            self.data.extend(f.read_text().strip().split("\n"))

        self.total_tokens = sum(len(line.split()) for line in self.data)
        self.total_sequences = len(self.data) // seq_len

    def __len__(self) -> int:
        return max(1, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> dict:
        line = self.data[idx]
        tokens = line.split()
        if len(tokens) < self.seq_len:
            tokens = tokens + ["<|pad|>"] * (self.seq_len - len(tokens))
        tokens = tokens[: self.seq_len]

        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        targets = torch.zeros(self.seq_len, dtype=torch.long)

        for i, token in enumerate(tokens):
            input_ids[i] = hash(token) % 32000
            if i > 0:
                targets[i] = input_ids[i - 1]
        targets[0] = -100

        return {"input_ids": input_ids, "labels": targets}


class ChatDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 2048, split: str = "train"):
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob(f"*.{split}.*"))

        self.data = []
        for f in self.files:
            import json
            with open(f) as fh:
                for line in fh:
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        messages = self.data[idx].get("messages", [])

        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        labels = torch.full((self.seq_len,), -100, dtype=torch.long)

        pos = 0
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            role_id = {"system": 3, "user": 4, "assistant": 5}.get(role, 4)

            if pos < self.seq_len:
                input_ids[pos] = role_id
                pos += 1

            for ch in content:
                if pos < self.seq_len:
                    input_ids[pos] = hash(ch) % 32000
                    if role == "assistant":
                        labels[pos] = input_ids[pos]
                    pos += 1

        return {"input_ids": input_ids[: self.seq_len], "labels": labels[: self.seq_len]}


class StreamingDataset(IterableDataset):
    def __init__(self, data_dir: str, seq_len: int = 2048):
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)

    def __iter__(self):
        files = sorted(self.data_dir.glob("*.jsonl"))
        for f in files:
            import json
            with open(f) as fh:
                buffer = []
                for line in fh:
                    try:
                        data = json.loads(line)
                        buffer.append(data)
                        if len(buffer) >= self.seq_len:
                            yield self._process(buffer[: self.seq_len])
                            buffer = buffer[self.seq_len // 2:]
                    except json.JSONDecodeError:
                        continue

    def _process(self, messages: list) -> dict:
        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        labels = torch.full((self.seq_len,), -100, dtype=torch.long)

        pos = 0
        for msg in messages:
            content = msg.get("text", msg.get("content", ""))
            for ch in content:
                if pos < self.seq_len:
                    input_ids[pos] = hash(ch) % 32000
                    labels[pos] = input_ids[pos]
                    pos += 1

        return {"input_ids": input_ids, "labels": labels}


def find_datasets(base_path: str) -> List[str]:
    base = Path(base_path)
    datasets = []
    if base.exists():
        for item in base.iterdir():
            if item.is_dir():
                has_data = any(item.glob("*.jsonl")) or any(item.glob("*.json")) or any(item.glob("*.parquet"))
                if has_data:
                    datasets.append(str(item))
    return datasets


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 4,
    distributed: bool = False,
) -> DataLoader:
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )
