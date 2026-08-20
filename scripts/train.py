#!/usr/bin/env python3
"""Seto training script — standalone CLI entry point."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto import SetoLM, SetoTokenizer, SetoTrainer, ModelConfig, TrainingConfig
from seto.data import PretrainDataset, ChatDataset, StreamingDataset, find_datasets
from seto.checkpoint import get_latest_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train Seto language model")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--model-config", type=str, default="small", choices=["small", "base"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="seto-output")
    parser.add_argument("--tokenizer", type=str, default="seto-tokenizer")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    model_config = ModelConfig()
    train_config = TrainingConfig()

    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.lr:
        train_config.lr = args.lr
    if args.max_steps:
        train_config.max_steps = args.max_steps
    train_config.use_bf16 = args.bf16
    train_config.local_rank = args.local_rank
    train_config.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    train_config.output_dir = args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"model": model_config.__dict__, "training": train_config.__dict__}, f, indent=2)

    print(f"Seto-1B | Params: ~{model_config.num_params():,}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")

    model = SetoLM(model_config)
    tokenizer = SetoTokenizer.from_pretrained(args.tokenizer)

    data_files = find_datasets(args.data_dir)
    if not data_files:
        print(f"No datasets found in {args.data_dir}")
        sys.exit(1)
    print(f"Found datasets: {data_files}")

    train_dataset = StreamingDataset(data_files[0], seq_len=model_config.max_seq_len)
    val_dataset = None
    if args.val_dir:
        val_files = find_datasets(args.val_dir)
        if val_files:
            val_dataset = StreamingDataset(val_files[0], seq_len=model_config.max_seq_len)

    trainer = SetoTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        local_rank=args.local_rank,
    )

    if args.resume:
        trainer.resume(args.resume)
    else:
        latest = get_latest_checkpoint(train_config.checkpoint_dir)
        if latest:
            print(f"Found checkpoint: {latest}")
            trainer.resume(latest)

    trainer.train()

    if args.local_rank in [-1, 0]:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state_dict, os.path.join(final_path, "model.pt"))
        print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
