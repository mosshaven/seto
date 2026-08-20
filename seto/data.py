"""Seto dataset loading — multilingual pretrain, SFT, DPO."""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from .data_filter import filter_text, detect_language


class PretrainDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 2048, tokenizer=None):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data_dir = Path(data_dir)

        self.files = []
        for ext in ["*.jsonl", "*.txt", "*.json"]:
            self.files.extend(self.data_dir.rglob(ext))

        self.data = []
        for f in self.files:
            try:
                if f.suffix == ".jsonl":
                    with open(f) as fh:
                        for line in fh:
                            try:
                                obj = json.loads(line)
                                text = obj.get("text", obj.get("content", ""))
                                if text:
                                    self.data.append(text)
                            except json.JSONDecodeError:
                                continue
                elif f.suffix == ".txt":
                    text = f.read_text(errors="ignore")
                    self.data.extend(text.split("\n\n"))
                elif f.suffix == ".json":
                    with open(f) as fh:
                        obj = json.load(fh)
                        if isinstance(obj, list):
                            for item in obj:
                                text = item.get("text", item.get("content", ""))
                                if text:
                                    self.data.append(text)
            except Exception:
                continue

        # Filter by quality and language
        filtered = []
        for text in self.data:
            passed, quality, reason = filter_text(
                text, min_quality=0.3, allowed_languages=["en", "ru", "uk"]
            )
            if passed and len(text) > 100:
                filtered.append(text)
        self.data = filtered

        print(f"Loaded {len(self.data)} texts from {data_dir}")

    def __len__(self) -> int:
        return max(1, len(self.data) - 1)

    def __getitem__(self, idx: int) -> dict:
        text = self.data[idx]

        if self.tokenizer:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) < self.seq_len:
                ids = ids + [0] * (self.seq_len - len(ids))
            else:
                ids = ids[:self.seq_len]
            input_ids = torch.tensor(ids, dtype=torch.long)
        else:
            input_ids = torch.zeros(self.seq_len, dtype=torch.long)
            for i, ch in enumerate(text[:self.seq_len]):
                input_ids[i] = ord(ch) % 32000

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 2048, tokenizer=None, max_samples: int = 500000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = []

        data_path = Path(data_dir)
        files = list(data_path.rglob("*.jsonl")) + list(data_path.rglob("*.json"))

        for f in files:
            try:
                if f.suffix == ".jsonl":
                    with open(f) as fh:
                        for line in fh:
                            try:
                                obj = json.loads(line)
                                messages = obj.get("messages", obj.get("conversations", []))
                                if messages:
                                    self.data.append(messages)
                            except json.JSONDecodeError:
                                continue
                elif f.suffix == ".json":
                    with open(f) as fh:
                        obj = json.load(fh)
                        if isinstance(obj, list):
                            for item in obj:
                                messages = item.get("messages", item.get("conversations", []))
                                if messages:
                                    self.data.append(messages)
            except Exception:
                continue

            if len(self.data) >= max_samples:
                break

        print(f"Loaded {len(self.data)} SFT samples from {data_dir}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        messages = self.data[idx]

        if self.tokenizer:
            text = self.tokenizer.apply_chat_template(messages)
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        else:
            ids = []
            for msg in messages:
                content = msg.get("content", "")
                for ch in content:
                    ids.append(ord(ch) % 32000)

        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        else:
            ids = ids[:self.seq_len]

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()

        # Mask non-assistant tokens for selective loss
        in_assistant = False
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                in_assistant = True
            elif msg.get("role") in ("system", "user"):
                in_assistant = False

            content = msg.get("content", "")
            content_len = len(content)

        # Simple approach: mask first half (prompt), train on second half (response)
        mask_len = len(ids) // 3
        labels[:mask_len] = -100

        return {"input_ids": input_ids, "labels": labels}


class DPODataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 2048, tokenizer=None, max_samples: int = 100000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = []

        data_path = Path(data_dir)
        files = list(data_path.rglob("*.jsonl")) + list(data_path.rglob("*.json"))

        for f in files:
            try:
                if f.suffix == ".jsonl":
                    with open(f) as fh:
                        for line in fh:
                            try:
                                obj = json.loads(line)
                                if "chosen" in obj and "rejected" in obj:
                                    self.data.append(obj)
                                elif "prompt" in obj and "chosen" in obj:
                                    self.data.append(obj)
                            except json.JSONDecodeError:
                                continue
                elif f.suffix == ".json":
                    with open(f) as fh:
                        obj = json.load(fh)
                        if isinstance(obj, list):
                            for item in obj:
                                if "chosen" in item and "rejected" in item:
                                    self.data.append(item)
            except Exception:
                continue

            if len(self.data) >= max_samples:
                break

        print(f"Loaded {len(self.data)} DPO pairs from {data_dir}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        pair = self.data[idx]
        prompt = pair.get("prompt", "")
        chosen = pair.get("chosen", "")
        rejected = pair.get("rejected", "")

        if self.tokenizer:
            prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            chosen_ids = self.tokenizer.encode(chosen, add_bos=False, add_eos=True)
            rejected_ids = self.tokenizer.encode(rejected, add_bos=False, add_eos=True)
        else:
            prompt_ids = [ord(c) % 32000 for c in prompt]
            chosen_ids = [ord(c) % 32000 for c in chosen] + [2]
            rejected_ids = [ord(c) % 32000 for c in rejected] + [2]

        # Truncate
        max_prompt = self.seq_len // 3
        prompt_ids = prompt_ids[:max_prompt]
        chosen_ids = chosen_ids[:self.seq_len - len(prompt_ids)]
        rejected_ids = rejected_ids[:self.seq_len - len(prompt_ids)]

        chosen_ids = prompt_ids + chosen_ids
        rejected_ids = prompt_ids + rejected_ids

        # Pad
        chosen_ids = chosen_ids + [0] * (self.seq_len - len(chosen_ids))
        rejected_ids = rejected_ids + [0] * (self.seq_len - len(rejected_ids))

        return {
            "input_ids": torch.tensor(chosen_ids[:self.seq_len], dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids[:self.seq_len], dtype=torch.long),
            "prompt_len": len(prompt_ids),
        }


class StreamingPretrainDataset(IterableDataset):
    def __init__(self, data_dir: str, seq_len: int = 2048, tokenizer=None):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data_dir = Path(data_dir)

    def __iter__(self):
        files = list(self.data_dir.rglob("*.jsonl")) + list(self.data_dir.rglob("*.txt"))
        for f in files:
            try:
                if f.suffix == ".jsonl":
                    with open(f) as fh:
                        for line in fh:
                            try:
                                obj = json.loads(line)
                                text = obj.get("text", obj.get("content", ""))
                                if text and len(text) > 100:
                                    passed, _, _ = filter_text(text, 0.3, ["en", "ru"])
                                    if passed:
                                        yield self._tokenize(text)
                            except json.JSONDecodeError:
                                continue
                elif f.suffix == ".txt":
                    text = f.read_text(errors="ignore")
                    for para in text.split("\n\n"):
                        if len(para) > 100:
                            passed, _, _ = filter_text(para, 0.3, ["en", "ru"])
                            if passed:
                                yield self._tokenize(para)
            except Exception:
                continue

    def _tokenize(self, text: str) -> dict:
        if self.tokenizer:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        else:
            ids = [ord(c) % 32000 for c in text]

        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        else:
            ids = ids[:self.seq_len]

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        return {"input_ids": input_ids, "labels": labels}


def find_datasets(base_path: str) -> List[str]:
    base = Path(base_path)
    datasets = []
    if base.exists():
        for item in base.iterdir():
            if item.is_dir():
                has_data = any(item.rglob("*.jsonl")) or any(item.rglob("*.json")) or any(item.rglob("*.txt"))
                if has_data:
                    datasets.append(str(item))
    return datasets


def create_dataloader(
    dataset,
    batch_size: int = 4,
    num_workers: int = 4,
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
