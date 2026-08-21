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
    MODEL_TINY, MODEL_SMALL, MODEL_BASE,
    STAGE_PRETRAIN, STAGE_COOLDOWN, STAGE_SFT, STAGE_DPO,
    SetoTrainer, SFTTrainer, DPOTrainer,
    ShardDataset, SFTDataset, DPODataset,
    get_latest_checkpoint,
)
from seto.trainer import setup_distributed, cleanup_distributed


def parse_args():
    p = argparse.ArgumentParser(description="Train Seto")
    p.add_argument("--stage", required=True, choices=["pretrain", "cooldown", "sft", "dpo"])
    p.add_argument("--model-config", default="tiny", choices=["tiny", "small", "base"])
    p.add_argument("--data-dir", required=True, help="Directory with .bin shards or SFT/DPO data")
    p.add_argument("--output-dir", default="seto-output")
    p.add_argument("--tokenizer", default="seto-tokenizer")
    p.add_argument("--resume", default=None, help="Resume from checkpoint (same stage)")
    p.add_argument("--init-from", default=None, help="Load model weights from previous stage")
    p.add_argument("--local-rank", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None, help="Override model max_seq_len")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", action="store_false", dest="fp16")
    p.add_argument("--dpo-ref-model", default=None)
    return p.parse_args()


def main():
    # Force unbuffered output so torchrun prints immediately
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

    args = parse_args()
    print(f"[train.py] Starting stage={args.stage} model={args.model_config}", flush=True)

    # Handle torchrun LOCAL_RANK
    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # Setup DDP
    setup_distributed(args.local_rank)
    is_main = args.local_rank in [-1, 0]

    try:
        model_map = {"tiny": MODEL_TINY, "small": MODEL_SMALL, "base": MODEL_BASE}
        model_config = model_map[args.model_config]

        stage_map = {
            "pretrain": STAGE_PRETRAIN,
            "cooldown": STAGE_COOLDOWN,
            "sft": STAGE_SFT,
            "dpo": STAGE_DPO,
        }
        train_config = stage_map[args.stage]

        # Set DDP-aware config values
        import torch.distributed as dist
        if dist.is_initialized():
            train_config.world_size = dist.get_world_size()

        # Sync max_seq_len from model config
        train_config.max_seq_len = model_config.max_seq_len
        if args.seq_len:
            train_config.max_seq_len = args.seq_len

        if args.batch_size:
            train_config.batch_size = args.batch_size
        if args.grad_accum:
            train_config.grad_accum_steps = args.grad_accum
        if args.lr:
            train_config.lr = args.lr
        if args.max_steps:
            train_config.max_steps = args.max_steps
        if args.save_every:
            train_config.save_every = args.save_every
        train_config.use_fp16 = args.fp16
        train_config.local_rank = args.local_rank
        train_config.checkpoint_dir = os.path.join(args.output_dir, f"checkpoints_{args.stage}")
        train_config.data_dir = args.data_dir

        if is_main:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, f"config_{args.stage}.json"), "w") as f:
                json.dump({"model": model_config.__dict__, "training": train_config.__dict__}, f, indent=2)
            print(f"Seto | Stage: {args.stage} | Model: {args.model_config} | Params: ~{model_config.num_params():,}")

        tokenizer = SetoTokenizer.from_pretrained(args.tokenizer)
        model = SetoLM(model_config)

        if is_main:
            print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6:.1f} MB")

        if args.stage in ("pretrain", "cooldown"):
            dataset = ShardDataset(args.data_dir, seq_len=model_config.max_seq_len)
            trainer = SetoTrainer(model, dataset, config=train_config, local_rank=args.local_rank)

            if args.resume:
                trainer.resume(args.resume)
            elif args.init_from:
                trainer.init_from(args.init_from)
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
            elif args.init_from:
                trainer.init_from(args.init_from)
            else:
                latest = get_latest_checkpoint(train_config.checkpoint_dir)
                if latest:
                    trainer.resume(latest)

            trainer.train()

        elif args.stage == "dpo":
            dataset = DPODataset(args.data_dir, seq_len=model_config.max_seq_len, tokenizer=tokenizer,
                                 max_samples=train_config.dpo_max_samples)

            import copy
            ref_model = SetoLM(model_config)
            if args.dpo_ref_model:
                from seto.checkpoint import load_checkpoint
                load_checkpoint(args.dpo_ref_model, ref_model)
            elif args.init_from:
                from seto.checkpoint import load_checkpoint
                load_checkpoint(args.init_from, ref_model)

            trainer = DPOTrainer(model, ref_model, dataset, tokenizer, train_config, args.local_rank)

            if args.resume:
                trainer.resume(args.resume)
            elif args.init_from:
                trainer.init_from(args.init_from)
            else:
                latest = get_latest_checkpoint(train_config.checkpoint_dir)
                if latest:
                    trainer.resume(latest)

            trainer.train()

        if is_main:
            final_dir = os.path.join(args.output_dir, f"final_{args.stage}")
            os.makedirs(final_dir, exist_ok=True)
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, os.path.join(final_dir, "model.pt"))
            with open(os.path.join(final_dir, "config.json"), "w") as f:
                json.dump(model_config.__dict__, f, indent=2)
            print(f"Saved to {final_dir}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
