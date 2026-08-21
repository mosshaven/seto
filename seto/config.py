"""Seto model and training configuration."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    vocab_size: int = 48000
    d_model: int = 1024
    n_layers: int = 14
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 2816
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    use_gradient_checkpointing: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def num_params(self) -> int:
        embed = self.vocab_size * self.d_model  # token embeddings
        per_layer = (
            self.d_model * (self.d_model + 2 * (self.d_model // self.n_heads * self.n_kv_heads))
            + self.d_model * self.d_model
            + 3 * self.d_model * self.d_ff
            + 2 * self.d_model  # 2x RMSNorm (attention_norm, ffn_norm)
        )
        final_norm = self.d_model  # final RMSNorm
        lm_head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return embed + self.n_layers * per_layer + final_norm + lm_head


MODEL_TINY = ModelConfig(
    vocab_size=48000,
    d_model=1024, n_layers=14, n_heads=16, n_kv_heads=4,
    d_ff=2816, max_seq_len=1024,
)

MODEL_SMALL = ModelConfig(
    vocab_size=48000,
    d_model=1280, n_layers=24, n_heads=20, n_kv_heads=5,
    d_ff=3584, max_seq_len=2048,
)

MODEL_BASE = ModelConfig(
    vocab_size=48000,
    d_model=2048, n_layers=22, n_heads=16, n_kv_heads=4,
    d_ff=5504, max_seq_len=2048,
)


@dataclass
class TrainConfig:
    stage: str = "pretrain"

    # Optimization
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 100000
    lr_schedule: str = "cosine"

    # Batch
    batch_size: int = 8
    grad_accum_steps: int = 4

    # Precision — FP16 for T4 (no bf16 on Turing)
    use_fp16: bool = True
    use_bf16: bool = False

    # Checkpointing
    save_every: int = 1000
    eval_every: int = 500
    keep_last_n: int = 1
    checkpoint_dir: str = "checkpoints"
    # Checkpoint: keep 1 dir, ZIP only for final export

    # Logging
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "seto"

    # Data
    data_dir: str = "data/shards"
    val_data: str = ""
    tokenizer_path: str = "seto-tokenizer"
    num_workers: int = 2
    max_seq_len: int = 1024

    # SFT config
    sft_data: str = ""
    sft_max_samples: int = 100000
    sft_loss_all_tokens: bool = False

    # DPO config
    dpo_data: str = ""
    dpo_beta: float = 0.1
    dpo_max_samples: int = 100000

    # DDP
    local_rank: int = -1
    world_size: int = 1

    # Resume / init
    resume_from: Optional[str] = None
    init_from: Optional[str] = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps * max(1, self.world_size)

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.max_seq_len


STAGE_PRETRAIN = TrainConfig(
    stage="pretrain",
    lr=3e-4, min_lr=3e-5,
    warmup_steps=2000, max_steps=100000,
    batch_size=8, grad_accum_steps=4,
    save_every=1000, eval_every=500,
)

STAGE_COOLDOWN = TrainConfig(
    stage="cooldown",
    lr=1e-4, min_lr=1e-5,
    warmup_steps=100, max_steps=10000,
    batch_size=8, grad_accum_steps=4,
    save_every=500, eval_every=250,
)

STAGE_SFT = TrainConfig(
    stage="sft",
    lr=5e-5, min_lr=5e-6,
    warmup_steps=200, max_steps=5000,
    batch_size=8, grad_accum_steps=4,
    save_every=500, eval_every=100,
    sft_loss_all_tokens=False,
)

STAGE_DPO = TrainConfig(
    stage="dpo",
    lr=5e-6, min_lr=5e-7,
    warmup_steps=100, max_steps=2000,
    batch_size=8, grad_accum_steps=4,
    save_every=500, eval_every=100,
    dpo_beta=0.1,
)
