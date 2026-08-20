"""Seto model and training configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 2048
    n_layers: int = 22
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 5504
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_head_dim(self) -> int:
        return self.d_model // self.n_heads

    def num_params(self) -> int:
        embed = self.vocab_size * self.d_model
        per_layer = (
            self.d_model * (self.d_model + 2 * (self.d_model // self.n_heads * self.n_kv_heads))
            + self.d_model * self.d_model
            + 3 * self.d_model * self.d_ff
        )
        total = embed + self.n_layers * per_layer + self.d_model
        if self.tie_embeddings:
            total -= self.vocab_size * self.d_model
        return total


@dataclass
class TrainingConfig:
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
    batch_size: int = 4
    grad_accum_steps: int = 8

    # Mixed precision
    use_bf16: bool = True

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Checkpointing
    save_every: int = 1000
    eval_every: int = 500
    keep_last_n: int = 3
    checkpoint_dir: str = "checkpoints"
    zip_checkpoints: bool = True

    # Logging
    log_every: int = 10
    wandb_project: str = "seto"
    wandb_entity: Optional[str] = None
    use_wandb: bool = False

    # Data
    train_datasets: list = field(default_factory=lambda: [
        "/kaggle/input/fineweb-edu-sample-10bt/fineweb_edu_sample_10B",
    ])
    val_datasets: list = field(default_factory=lambda: [
        "/kaggle/input/dataset-name/path",
    ])
    tokenizer_path: str = "seto-tokenizer"
    num_workers: int = 4
    streaming: bool = False

    # DDP
    local_rank: int = -1
    world_size: int = 1

    # Resume
    resume_from: Optional[str] = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps * max(1, self.world_size)

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * 512  # avg seq len estimate
