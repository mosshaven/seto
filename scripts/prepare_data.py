#!/usr/bin/env python3
"""Seto data preparation — download, tokenize, pack into uint16 shards."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto.tokenizer import SetoTokenizer
from seto.data import pack_from_hf_dataset


def prepare_tokenizer(output_dir: str, vocab_size: int = 48000):
    """Train tokenizer on multilingual data."""
    print(f"Training tokenizer (vocab={vocab_size})...")

    from datasets import load_dataset

    samples = []

    # Russian from FineWeb2 — 5000 docs
    print("  Sampling Russian text...")
    try:
        ru_ds = load_dataset("HuggingFaceFW/fineweb-2", name="rus_Cyrl", split="train", streaming=True)
        for i, row in enumerate(ru_ds):
            if i >= 5000:
                break
            text = row.get("text", "")
            if text and len(text) > 100:
                samples.append(text)
    except Exception as e:
        print(f"  Warning: Could not load FineWeb2 RU: {e}")

    # Ukrainian — 500 docs
    print("  Sampling Ukrainian text...")
    try:
        uk_ds = load_dataset("HuggingFaceFW/fineweb-2", name="ukr_Cyrl", split="train", streaming=True)
        for i, row in enumerate(uk_ds):
            if i >= 500:
                break
            text = row.get("text", "")
            if text and len(text) > 100:
                samples.append(text)
    except Exception as e:
        print(f"  Warning: Could not load FineWeb2 UK: {e}")

    # English — 500 docs
    print("  Sampling English text...")
    try:
        en_ds = load_dataset("HuggingFaceFW/fineweb_100BT", split="train", streaming=True)
        for i, row in enumerate(en_ds):
            if i >= 500:
                break
            text = row.get("text", "")
            if text and len(text) > 100:
                samples.append(text)
    except Exception as e:
        print(f"  Warning: Could not load FineWeb EN: {e}")

    if not samples:
        print("ERROR: No samples collected for tokenizer training")
        return

    # Write samples to temp file
    temp_file = os.path.join(output_dir, "tokenizer_samples.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(samples))

    # Train tokenizer
    tokenizer = SetoTokenizer(vocab_size=vocab_size)
    tokenizer.train([temp_file], output_dir)

    # Cleanup
    os.remove(temp_file)

    print(f"Tokenizer trained: {len(tokenizer)} vocab")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare Seto training data")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--tokenizer-dir", default="seto-tokenizer", help="Tokenizer output dir")
    parser.add_argument("--vocab-size", type=int, default=48000)
    parser.add_argument("--max-samples-ru", type=int, default=100000,
                        help="Max Russian samples from FineWeb2")
    parser.add_argument("--max-samples-wiki", type=int, default=20000,
                        help="Max Wikipedia samples")
    parser.add_argument("--shard-size", type=int, default=100_000_000,
                        help="Tokens per shard")
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    shard_dir = os.path.join(args.output_dir, "shards")

    if not args.skip_tokenizer:
        tokenizer = prepare_tokenizer(args.tokenizer_dir, args.vocab_size)
    else:
        tokenizer = SetoTokenizer.from_pretrained(args.tokenizer_dir)

    # FineWeb2 Russian — 85%
    print("Packing FineWeb2 Russian...")
    pack_from_hf_dataset(
        "HuggingFaceFW/fineweb-2",
        tokenizer, shard_dir,
        text_key="text",
        max_samples=args.max_samples_ru,
        shard_size=args.shard_size,
        split="train",
        config_name="rus_Cyrl",
    )

    # Wikipedia Russian — 15%
    print("Packing Wikipedia Russian...")
    pack_from_hf_dataset(
        "wikimedia/wikipedia",
        tokenizer, shard_dir,
        text_key="text",
        max_samples=args.max_samples_wiki,
        shard_size=args.shard_size,
        split="train",
        config_name="20231101.ru",
    )

    print(f"\nDone! Data ready at {shard_dir}")
    print(f"Tokenizer at {args.tokenizer_dir}")


if __name__ == "__main__":
    main()
