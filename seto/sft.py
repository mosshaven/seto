"""Seto SFT — supervised fine-tuning with chat template."""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .checkpoint import save_checkpoint, get_latest_checkpoint
from .config import TrainConfig


class SFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        tokenizer,
        config: TrainConfig,
        local_rank: int = -1,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.is_main = local_rank in [-1, 0]
        self.device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cpu")

        model = model.to(self.device)
        if local_rank >= 0:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        self.model = model

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        self.use_amp = (config.use_fp16 or config.use_bf16) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if config.use_bf16 else torch.float16
        self.scaler = GradScaler(enabled=self.use_amp and config.use_fp16)
        self.global_step = 0

        from torch.utils.data import DataLoader
        sampler = None
        if local_rank >= 0 and hasattr(dataset, '__len__'):
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(sampler is None and hasattr(dataset, '__len__')),
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )
        self.sampler = sampler

    def resume(self, checkpoint_path: str):
        meta = load_checkpoint(
            checkpoint_path, self.model.module if hasattr(self.model, "module") else self.model,
            self.optimizer, str(self.device)
        )
        self.global_step = meta.get("step", 0)
        for _ in range(self.global_step):
            self.scheduler.step()
        if self.is_main:
            print(f"Resumed from step {self.global_step}")

    def train(self):
        if self.is_main:
            print(f"SFT Training | Steps: {self.config.max_steps} | LR: {self.config.lr}")
            m = self.model.module if hasattr(self.model, "module") else self.model
            print(f"Model params: {m.count_parameters():,}")

        self.model.train()
        running_loss = 0.0
        start_time = time.time()

        while self.global_step < self.config.max_steps:
            if self.sampler is not None:
                self.sampler.set_epoch(self.global_step)

            for batch_idx, batch in enumerate(self.data_loader):
                if self.global_step >= self.config.max_steps:
                    break

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    logits, _ = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    ) / self.config.grad_accum_steps

                if self.config.use_fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item()

                if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                    if self.config.use_fp16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    if self.config.use_fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.is_main and self.global_step % self.config.log_every == 0:
                        elapsed = time.time() - start_time
                        avg_loss = running_loss / self.config.log_every
                        print(f"  SFT Step {self.global_step:>5d} | Loss {avg_loss:.4f} | {elapsed:.1f}s")
                        running_loss = 0.0
                        start_time = time.time()

                    if self.global_step % self.config.save_every == 0 and self.is_main:
                        self._save()

    def _save(self):
        save_checkpoint(
            model=self.model.module if hasattr(self.model, "module") else self.model,
            optimizer=self.optimizer,
            step=self.global_step,
            loss=0.0,
            config={"stage": "sft", "step": self.global_step},
            save_dir=self.config.checkpoint_dir,
            zip_it=self.config.zip_checkpoints,
            keep_last_n=self.config.keep_last_n,
            tokens_seen=self.global_step * self.config.tokens_per_step,
        )
