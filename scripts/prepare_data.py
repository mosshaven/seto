#!/usr/bin/env python3
"""Seto data preparation — download, tokenize, pack into uint16 shards."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto.tokenizer import SetoTokenizer
from seto.data import pack_from_hf_dataset, pack_tokens_to_shards


def prepare_tokenizer(output_dir: str, vocab_size: int = 48000):
    """Train tokenizer on multilingual data."""
    print(f"Training tokenizer (vocab={vocab_size})...")

    # Download sample text for tokenizer training
    from datasets import load_dataset

    samples = []

    # Russian from FineWeb2
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

    # Ukrainian
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

    # English
    print("  Sampling English text...")
    try:
        en_ds = load_dataset("HuggingFaceFW/fineweb-100BT", split="train", streaming=True)
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


def prepare_shards(
    output_dir: str,
    tokenizer,
    max_samples_per_dataset: int = 100000,
    shard_size: int = 100_000_000,
):
    """Download datasets and pack into shards."""
    from datasets import load_dataset

    all_tokens = []

    # FineWeb2 Russian
    print("Loading FineWeb2 Russian...")
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-2", name="rus_Cyrl", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= max_samples_per_dataset:
                break
            text = row.get("text", "")
            if text and len(text) > 100:
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                all_tokens.extend(ids)
                count += 1
            if count % 10000 == 0 and count > 0:
                print(f"  FineWeb2 RU: {count:,} samples, {len(all_tokens):,} tokens")
    except Exception as e:
        print(f"  Warning: {e}")

    # Wikipedia Russian
    print("Loading Wikipedia Russian...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= max_samples_per_dataset // 5:
                break
            text = row.get("text", "")
            if text and len(text) > 100:
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                all_tokens.extend(ids)
                count += 1
            if count % 5000 == 0 and count > 0:
                print(f"  Wiki RU: {count:,} samples, {len(all_tokens):,} tokens")
    except Exception as e:
        print(f"  Warning: {e}")

    print(f"Total tokens: {len(all_tokens):,}")

    # Pack into shards
    pack_tokens_to_shards(all_tokens, output_dir, shard_size, split="train")


def main():
    parser = argparse.ArgumentParser(description="Prepare Seto training data")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--tokenizer-dir", default="seto-tokenizer", help="Tokenizer output dir")
    parser.add_argument("--vocab-size", type=int, default=48000)
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Max samples per dataset")
    parser.add_argument("--shard-size", type=int, default=100_000_000,
                        help="Tokens per shard")
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    shard_dir = os.path.join(args.output_dir, "shards")

    if not args.skip_tokenizer:
        tokenizer = prepare_tokenizer(args.tokenizer_dir, args.vocab_size)
    else:
        tokenizer = SetoTokenizer.from_pretrained(args.tokenizer_dir)

    prepare_shards(shard_dir, tokenizer, args.max_samples, args.shard_size)

    print(f"\nDone! Data ready at {shard_dir}")
    print(f"Tokenizer at {args.tokenizer_dir}")


if __name__ == "__main__":
    main()
