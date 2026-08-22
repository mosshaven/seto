#!/usr/bin/env python3
"""Validate SFT splitting and tool-call loss masking before training."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto.data import SFTDataset
from seto.tokenizer import SetoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = SetoTokenizer.from_pretrained(args.tokenizer)
    missing = [
        token
        for token in tokenizer.special_tokens.values()
        if tokenizer._tokenizer.token_to_id(token) is None
    ]
    if missing:
        tokenizer.add_special_tokens(missing)

    dataset = SFTDataset(args.dataset, args.seq_len, tokenizer)
    roles = [example["completion"]["role"] for example in dataset.examples]
    if "tool_call" not in roles:
        raise RuntimeError("dataset has no tool_call training target")

    index = roles.index("tool_call")
    example = dataset.examples[index]
    batch = dataset[index]
    labels = batch["labels"]
    input_ids = batch["input_ids"]

    prompt_text = tokenizer.apply_chat_template(
        example["prompt"], add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
    expected_target_position = len(prompt_ids) - 1
    if expected_target_position >= len(labels):
        raise RuntimeError("tool-call target was truncated from sequence")
    if not torch.all(labels[:expected_target_position] == -100):
        raise RuntimeError("prompt tokens contribute to SFT loss")
    if labels[expected_target_position].item() != tokenizer.tool_call_id:
        raise RuntimeError("first tool-call target is not <|tool_call|>")
    if torch.any(labels[input_ids == tokenizer.pad_id] != -100):
        raise RuntimeError("padding contributes to SFT loss")
    if torch.all(labels == -100):
        raise RuntimeError("tool-call example has no trainable labels")

    print(f"SFT examples: {len(dataset)}")
    print(f"Assistant targets: {roles.count('assistant')}")
    print(f"Tool-call targets: {roles.count('tool_call')}")
    print(f"Tool-call token id: {tokenizer.tool_call_id}")
    print("Prompt/tool/padding loss masking: OK")


if __name__ == "__main__":
    main()
