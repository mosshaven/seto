"""Seto DPO — Direct Preference Optimization."""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .checkpoint import save_checkpoint
from .config import TrainConfig


class DPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
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
        self.beta = config.dpo_beta

        model = model.to(self.device)
        ref_model = ref_model.to(self.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        if local_rank >= 0:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        self.model = model
        self.ref_model = ref_model

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
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

    def _get_log_probs(self, model, input_ids, prompt_len):
        """Get log probabilities for completion tokens only (after prompt)."""
        logits, _ = model(input_ids)
        # Shift: logits at t predict token at t+1
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(2)).squeeze(2)

        # Mask prompt tokens (only count completion)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, prompt_len - 1:] = True  # prompt_len-1 because of shift
        token_log_probs = token_log_probs * mask.float()

        # Sum over completion tokens
        return token_log_probs.sum(dim=-1)

    def dpo_loss(self, chosen_logps, rejected_logps):
        logits = self.beta * (chosen_logps - rejected_logps)
        loss = -F.logsigmoid(logits).mean()
        reward_accuracy = (logits > 0).float().mean()
        reward_margin = (chosen_logps - rejected_logps).mean()
        return loss, reward_accuracy, reward_margin

    def train(self):
        if self.is_main:
            print(f"DPO Training | Steps: {self.config.max_steps} | Beta: {self.beta}")

        self.model.train()
        running_loss = 0.0
        start_time = time.time()

        while self.global_step < self.config.max_steps:
            if self.sampler is not None:
                self.sampler.set_epoch(self.global_step)

            for batch_idx, batch in enumerate(self.data_loader):
                if self.global_step >= self.config.max_steps:
                    break

                chosen_ids = batch["input_ids"].to(self.device, non_blocking=True)
                rejected_ids = batch["rejected_ids"].to(self.device, non_blocking=True)
                prompt_len = batch["prompt_len"]

                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    chosen_logps = self._get_log_probs(self.model, chosen_ids, prompt_len)
                    rejected_logps = self._get_log_probs(self.model, rejected_ids, prompt_len)

                    with torch.no_grad():
                        ref_chosen_logps = self._get_log_probs(self.ref_model, chosen_ids, prompt_len)
                        ref_rejected_logps = self._get_log_probs(self.ref_model, rejected_ids, prompt_len)

                    chosen_logps = chosen_logps - ref_chosen_logps
                    rejected_logps = rejected_logps - ref_rejected_logps

                    loss, accuracy, margin = self.dpo_loss(chosen_logps, rejected_logps)
                    loss = loss / self.config.grad_accum_steps

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
                        print(
                            f"  DPO Step {self.global_step:>5d} | "
                            f"Loss {avg_loss:.4f} | Acc {accuracy:.2%} | "
                            f"Margin {margin:.4f} | {elapsed:.1f}s"
                        )
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
            config={"stage": "dpo", "step": self.global_step, "beta": self.beta},
            save_dir=self.config.checkpoint_dir,
            zip_it=self.config.zip_checkpoints,
            keep_last_n=self.config.keep_last_n,
            tokens_seen=self.global_step * self.config.tokens_per_step,
        )
