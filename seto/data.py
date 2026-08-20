"""Seto data — uint16 binary shards, SFT dataset, DPO dataset."""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


class ShardDataset(Dataset):
    """Memory-mapped uint16 binary shards — true mmap, no concatenation."""

    def __init__(self, data_dir: str, seq_len: int = 1024, split: str = "train"):
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)

        shard_files = sorted(self.data_dir.glob(f"{split}_*.bin"))
        if not shard_files:
            shard_files = sorted(self.data_dir.glob("*.bin"))

        # True memory-mapped: list of memmaps + cumulative offsets
        self.shards = []
        self.offsets = [0]  # cumulative token offsets
        total = 0
        for f in shard_files:
            mm = np.memmap(str(f), dtype=np.uint16, mode='r')
            self.shards.append(mm)
            total += len(mm)
            self.offsets.append(total)

        self.n_tokens = total
        self.n_sequences = self.n_tokens // (seq_len + 1)

        print(f"Loaded {self.n_tokens:,} tokens from {len(shard_files)} shards (true mmap)")

    def __len__(self) -> int:
        return max(1, self.n_sequences - 1)

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        # Gather tokens across shard boundaries
        chunk = self._gather(start, end)
        if len(chunk) < self.seq_len + 1:
            chunk = np.pad(chunk, (0, self.seq_len + 1 - len(chunk)), constant_values=0)

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}

    def _gather(self, start: int, end: int) -> np.ndarray:
        """Gather tokens from mmap shards without concatenation."""
        # Find which shards contain [start, end)
        parts = []
        for i in range(len(self.shards)):
            shard_start = self.offsets[i]
            shard_end = self.offsets[i + 1]
            if shard_end <= start or shard_start >= end:
                continue
            # Overlap of [start,end) with [shard_start,shard_end)
            s = max(start, shard_start) - shard_start
            e = min(end, shard_end) - shard_start
            parts.append(self.shards[i][s:e])
        return np.concatenate(parts) if parts else np.array([], dtype=np.uint16)


class SFTDataset(Dataset):
    """SFT dataset with chat template. Loss only on assistant turns."""

    def __init__(self, data_dir: str, seq_len: int = 1024, tokenizer=None, max_samples: int = 100000):
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
            # Prompt: all messages except last (assistant response)
            prompt_messages = messages[:-1]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True,
            )
            prompt_ids = self.tokenizer.encode(prompt_text, add_bos=True, add_eos=False)

            # Answer: last assistant message + EOS
            answer = messages[-1].get("content", "")
            answer_ids = self.tokenizer.encode(answer, add_bos=False, add_eos=True)

            # Full sequence: prompt + answer
            full_ids = prompt_ids + answer_ids
            prompt_len = len(prompt_ids)
        else:
            full_ids = []
            prompt_len = 0
            for i, msg in enumerate(messages):
                content = msg.get("content", "")
                for ch in content:
                    full_ids.append(ord(ch) % 48000)
                if i < len(messages) - 1:
                    prompt_len = len(full_ids)
            # Add EOS
            full_ids.append(2)

        # Pad/truncate
        if len(full_ids) < self.seq_len:
            full_ids = full_ids + [0] * (self.seq_len - len(full_ids))
        else:
            full_ids = full_ids[:self.seq_len]

        # Causal shift: input[:-1] predicts input[1:]
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        labels = torch.tensor(full_ids[1:], dtype=torch.long)

        # Mask prompt tokens and padding
        # After shift: label at position t corresponds to input token at t+1
        # So prompt tokens (positions 0..prompt_len-2 in labels) get -100
        labels[:prompt_len - 1] = -100
        labels[input_ids == 0] = -100

        return {"input_ids": input_ids, "labels": labels}


class DPODataset(Dataset):
    """DPO preference dataset with chosen/rejected pairs."""

    def __init__(self, data_dir: str, seq_len: int = 1024, tokenizer=None, max_samples: int = 100000):
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
            prompt_ids = [ord(c) % 48000 for c in prompt]
            chosen_ids = [ord(c) % 48000 for c in chosen] + [2]
            rejected_ids = [ord(c) % 48000 for c in rejected] + [2]

        max_prompt = self.seq_len // 3
        prompt_ids = prompt_ids[:max_prompt]
        chosen_ids = (prompt_ids + chosen_ids)[:self.seq_len]
        rejected_ids = (prompt_ids + rejected_ids)[:self.seq_len]

        prompt_len = len(prompt_ids)

        # Pad
        chosen_ids = chosen_ids + [0] * (self.seq_len - len(chosen_ids))
        rejected_ids = rejected_ids + [0] * (self.seq_len - len(rejected_ids))

        return {
            "input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "prompt_len": prompt_len,
        }


def pack_tokens_to_shards(
    token_ids: List[int],
    output_dir: str,
    shard_size: int = 100_000_000,
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
    """Read text files, tokenize, and pack into shards. Incremental — low RAM."""
    os.makedirs(output_dir, exist_ok=True)

    buffer = []
    shard_idx = 0
    total_tokens = 0

    for f in text_files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            buffer.extend(ids)

            # Write shard when buffer is full
            while len(buffer) >= shard_size:
                shard = np.array(buffer[:shard_size], dtype=np.uint16)
                shard_path = os.path.join(output_dir, f"{split}_{shard_idx:04d}.bin")
                shard.tofile(shard_path)
                print(f"  Shard {shard_idx}: {shard_size:,} tokens -> {shard_path}")
                buffer = buffer[shard_size:]
                shard_idx += 1
                total_tokens += shard_size

        except Exception as e:
            print(f"  Warning: failed to process {f}: {e}")

    # Write remaining buffer
    if buffer:
        shard = np.array(buffer, dtype=np.uint16)
        shard_path = os.path.join(output_dir, f"{split}_{shard_idx:04d}.bin")
        shard.tofile(shard_path)
        total_tokens += len(buffer)
        print(f"  Shard {shard_idx}: {len(buffer):,} tokens -> {shard_path}")

    print(f"Packed {total_tokens:,} tokens total")


def pack_from_hf_dataset(
    dataset_name: str,
    tokenizer,
    output_dir: str,
    text_key: str = "text",
    max_samples: Optional[int] = None,
    shard_size: int = 100_000_000,
    split: str = "train",
    config_name: Optional[str] = None,
    shard_start_idx: int = 0,
):
    """Load HuggingFace dataset, tokenize, and pack into shards. Incremental."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, name=config_name, split=split, streaming=True)

    os.makedirs(output_dir, exist_ok=True)
    buffer = []
    shard_idx = shard_start_idx
    total_tokens = 0
    count = 0

    for row in ds:
        if max_samples and count >= max_samples:
            break

        text = row.get(text_key, "")
        if text and len(text) > 100:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            buffer.extend(ids)
            count += 1

        # Write shard when buffer is full
        while len(buffer) >= shard_size:
            shard = np.array(buffer[:shard_size], dtype=np.uint16)
            shard_path = os.path.join(output_dir, f"{split}_{shard_idx:04d}.bin")
            shard.tofile(shard_path)
            print(f"  Shard {shard_idx}: {shard_size:,} tokens -> {shard_path}")
            buffer = buffer[shard_size:]
            shard_idx += 1
            total_tokens += shard_size

        if count % 10000 == 0 and count > 0:
            print(f"  Tokenized {count:,} samples, {total_tokens + len(buffer):,} tokens...")

    # Write remaining
    if buffer:
        shard = np.array(buffer, dtype=np.uint16)
        shard_path = os.path.join(output_dir, f"{split}_{shard_idx:04d}.bin")
        shard.tofile(shard_path)
        total_tokens += len(buffer)
        print(f"  Shard {shard_idx}: {len(buffer):,} tokens -> {shard_path}")

    print(f"Packed {total_tokens:,} tokens from {count:,} samples")


def find_shards(data_dir: str, split: str = "train") -> List[str]:
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
