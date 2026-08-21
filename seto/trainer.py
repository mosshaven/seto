"""Seto trainer — pretraining with FP16, SDPA, full checkpoints."""

import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from .config import ModelConfig, TrainConfig


def setup_distributed(local_rank: int):
    if local_rank == -1:
        return
    # torchrun already sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    # Only call init_process_group if torchrun didn't already do it
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_cosine_schedule(optimizer, warmup_steps: int, max_steps: int, min_lr: float, base_lr: float):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr / base_lr + 0.5 * (1.0 - min_lr / base_lr) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class SetoTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        config: Optional[TrainConfig] = None,
        local_rank: int = -1,
    ):
        self.config = config or TrainConfig()
        self.local_rank = local_rank
        self.is_main = local_rank in [-1, 0]

        if local_rank >= 0:
            self.device = torch.device(f"cuda:{local_rank}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        model = model.to(self.device)

        if local_rank >= 0:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        self.model = model

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        # FP16 for T4 (Turing) — bf16 not supported on T4
        use_amp = (self.config.use_fp16 or self.config.use_bf16) and self.device.type == "cuda"
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16 if self.config.use_bf16 else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp and self.config.use_fp16)

        self.scheduler = get_cosine_schedule(
            self.optimizer, self.config.warmup_steps, self.config.max_steps,
            self.config.min_lr, self.config.lr
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.global_step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")

        self._setup_dataloader()

    def _setup_dataloader(self):
        from torch.utils.data import DataLoader
        sampler = None
        if self.local_rank >= 0 and hasattr(self.train_dataset, '__len__'):
            sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None and hasattr(self.train_dataset, '__len__')),
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )
        self.train_sampler = sampler

        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        else:
            self.val_loader = None

    def _load_state(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load full checkpoint state."""
        meta = load_checkpoint(
            checkpoint_path, self.model.module if hasattr(self.model, "module") else self.model,
            self.optimizer if load_optimizer else None, str(self.device)
        )
        self.global_step = meta.get("step", 0)
        self.tokens_seen = meta.get("tokens_seen", 0)

        # Restore scheduler state
        if "scheduler" in meta and meta["scheduler"] is not None:
            self.scheduler.load_state_dict(meta["scheduler"])
        elif self.global_step > 0:
            # Fallback: step N times (less exact)
            for _ in range(self.global_step):
                self.scheduler.step()

        # Restore RNG state
        if "rng" in meta:
            rng = meta["rng"]
            random.setstate(rng.get("python"))
            np.random.set_state(rng.get("numpy"))
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng["cuda"])

        # Restore GradScaler
        if "scaler" in meta and meta["scaler"] is not None:
            self.scaler.load_state_dict(meta["scaler"])

        if self.is_main:
            print(f"Loaded from step {self.global_step} ({self.tokens_seen:,} tokens)")

    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint (model + optimizer + scheduler + RNG)."""
        self._load_state(checkpoint_path, load_optimizer=True)

    def init_from(self, checkpoint_path: str):
        """Load model weights only (for stage chaining: pretrain→cooldown→sft)."""
        self._load_state(checkpoint_path, load_optimizer=False)
        # Reset optimizer for new stage
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = get_cosine_schedule(
            self.optimizer, self.config.warmup_steps, self.config.max_steps,
            self.config.min_lr, self.config.lr
        )
        self.global_step = 0
        self.tokens_seen = 0
        if self.is_main:
            print(f"Initialized from {checkpoint_path} (model weights only)")

    def train(self):
        from contextlib import nullcontext
        if self.is_main:
            m = self.model.module if hasattr(self.model, "module") else self.model
            print(f"Pretraining | Steps: {self.config.max_steps} | LR: {self.config.lr}")
            print(f"Model params: {m.count_parameters():,}")
            print(f"Effective batch size: {self.config.effective_batch_size}")
            print(f"Tokens per step: ~{self.config.tokens_per_step:,}")
            print(f"FP16: {self.config.use_fp16}")

        self.model.train()
        running_loss = 0.0
        start_time = time.time()
        is_ddp = hasattr(self.model, "module") and hasattr(self.model, "no_sync")

        while self.global_step < self.config.max_steps:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.global_step)

            for batch_idx, batch in enumerate(self.train_loader):
                if self.global_step >= self.config.max_steps:
                    break

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    _, loss = self.model(input_ids, targets=labels)
                    loss = loss / self.config.grad_accum_steps

                # DDP: skip gradient sync on microbatches (except last)
                sync_now = (batch_idx + 1) % self.config.grad_accum_steps == 0
                ctx = nullcontext() if (not is_ddp or sync_now) else self.model.no_sync()
                with ctx:
                    if self.config.use_fp16:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                running_loss += loss.item()

                if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                    if self.config.use_fp16:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                    if self.config.use_fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1
                    self.tokens_seen += self.config.tokens_per_step

                    if self.is_main and self.global_step % self.config.log_every == 0:
                        elapsed = time.time() - start_time
                        avg_loss = running_loss / self.config.log_every
                        lr = self.scheduler.get_last_lr()[0]
                        tokens_sec = self.config.tokens_per_step / elapsed * self.config.log_every
                        print(
                            f"Step {self.global_step:>6d} | "
                            f"Loss {avg_loss:.4f} | "
                            f"LR {lr:.2e} | "
                            f"Tokens {self.tokens_seen:,} | "
                            f"{tokens_sec:,.0f} tok/s | "
                            f"{elapsed:.1f}s"
                        )
                        running_loss = 0.0
                        start_time = time.time()

                    if self.global_step % self.config.eval_every == 0 and self.val_loader is not None:
                        val_loss = self.evaluate()
                        if self.is_main:
                            print(f"  Eval @ step {self.global_step}: val_loss={val_loss:.4f}")
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self._save_checkpoint(val_loss, is_best=True)

                    if self.global_step % self.config.save_every == 0 and self.is_main:
                        self._save_checkpoint(running_loss / max(1, self.config.log_every))

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss = self.model(input_ids, targets=labels)
            total_loss += loss.item()
            n_batches += 1
        self.model.train()
        return total_loss / max(1, n_batches)

    def _save_checkpoint(self, loss: float, is_best: bool = False):
        # Save RNG state
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()

        config_dict = {
            "stage": self.config.stage,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
        }

        save_checkpoint(
            model=self.model.module if hasattr(self.model, "module") else self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.global_step,
            loss=loss,
            config=config_dict,
            save_dir=self.config.checkpoint_dir,
            zip_it=self.config.zip_checkpoints,
            keep_last_n=self.config.keep_last_n,
            rng_state=rng_state,
            tokens_seen=self.tokens_seen,
        )
        if is_best:
            best_dir = os.path.join(self.config.checkpoint_dir, "best")
            save_checkpoint(
                model=self.model.module if hasattr(self.model, "module") else self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=self.global_step,
                loss=loss,
                config=config_dict,
                save_dir=best_dir,
                zip_it=True,
                keep_last_n=1,
                rng_state=rng_state,
                tokens_seen=self.tokens_seen,
            )
