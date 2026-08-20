#!/usr/bin/env python3
"""Seto training script — pretrain / cooldown / sft / dpo."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto import (
    SetoLM, SetoTokenizer, ModelConfig, TrainConfig,
    STAGE_PRETRAIN, STAGE_COOLDOWN, STAGE_SFT, STAGE_DPO,
    SetoTrainer, SFTTrainer, DPOTrainer,
    PretrainDataset, SFTDataset, DPODataset,
    find_datasets, get_latest_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Seto")
    p.add_argument("--stage", required=True, choices=["pretrain", "cooldown", "sft", "dpo"])
    p.add_argument("--model-config", default="base", choices=["small", "base"])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--val-dir", default=None)
    p.add_argument("--output-dir", default="seto-output")
    p.add_argument("--tokenizer", default="seto-tokenizer")
    p.add_argument("--resume", default=None)
    p.add_argument("--local-rank", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_false", dest="bf16")
    # DPO specific
    p.add_argument("--ref-model", default=None, help="Path to reference model for DPO")
    p.add_argument("--dpo-beta", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()

    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # Load model config
    model_config = MODEL_SMALL if args.model_config == "small" else MODEL_BASE

    # Load stage config
    stage_map = {
        "pretrain": STAGE_PRETRAIN,
        "cooldown": STAGE_COOLDOWN,
        "sft": STAGE_SFT,
        "dpo": STAGE_DPO,
    }
    train_config = stage_map[args.stage]

    # Override from CLI
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.lr:
        train_config.lr = args.lr
    if args.max_steps:
        train_config.max_steps = args.max_steps
    train_config.use_bf16 = args.bf16
    train_config.local_rank = args.local_rank
    train_config.checkpoint_dir = os.path.join(args.output_dir, f"checkpoints_{args.stage}")
    train_config.train_data = args.data_dir
    if args.val_dir:
        train_config.val_data = args.val_dir

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"config_{args.stage}.json"), "w") as f:
        json.dump({"model": model_config.__dict__, "training": train_config.__dict__}, f, indent=2)

    print(f"Seto | Stage: {args.stage} | Params: ~{model_config.num_params():,}")

    # Load tokenizer
    tokenizer = SetoTokenizer.from_pretrained(args.tokenizer)

    # Load model
    model = SetoLM(model_config)

    if args.stage in ("pretrain", "cooldown"):
        dataset = PretrainDataset(args.data_dir, seq_len=model_config.max_seq_len, tokenizer=tokenizer)
        val_dataset = None
        if args.val_dir:
            val_dataset = PretrainDataset(args.val_dir, seq_len=model_config.max_seq_len, tokenizer=tokenizer)

        trainer = SetoTrainer(model, dataset, val_dataset, train_config, args.local_rank)

        if args.resume:
            trainer.resume(args.resume)
        else:
            latest = get_latest_checkpoint(train_config.checkpoint_dir)
            if latest:
                trainer.resume(latest)

        trainer.train()

    elif args.stage == "sft":
        dataset = SFTDataset(args.data_dir, seq_len=model_config.max_seq_len, tokenizer=tokenizer,
                             max_samples=train_config.sft_max_samples)

        trainer = SFTTrainer(model, dataset, tokenizer, train_config, args.local_rank)

        if args.resume:
            trainer.resume(args.resume)
        else:
            latest = get_latest_checkpoint(train_config.checkpoint_dir)
            if latest:
                trainer.resume(latest)

        trainer.train()

    elif args.stage == "dpo":
        dataset = DPODataset(args.data_dir, seq_len=model_config.max_seq_len, tokenizer=tokenizer,
                             max_samples=train_config.dpo_max_samples)

        # Reference model is the SFT model
        ref_model = SetoLM(model_config)
        if args.ref_model:
            from seto.checkpoint import load_checkpoint
            load_checkpoint(args.ref_model, ref_model)

        trainer = DPOTrainer(model, ref_model, dataset, tokenizer, train_config, args.local_rank)

        if args.resume:
            trainer.resume(args.resume)
        else:
            latest = get_latest_checkpoint(train_config.checkpoint_dir)
            if latest:
                trainer.resume(latest)

        trainer.train()

    # Save final
    if args.local_rank in [-1, 0]:
        final_dir = os.path.join(args.output_dir, f"final_{args.stage}")
        os.makedirs(final_dir, exist_ok=True)
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state_dict, os.path.join(final_dir, "model.pt"))
        print(f"Done. Saved to {final_dir}")


if __name__ == "__main__":
    main()
