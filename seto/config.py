"""Seto model and training configuration — multi-stage pipeline."""

from dataclasses import dataclass, field
from typing import Optional, List


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


# Presets
MODEL_SMALL = ModelConfig(
    d_model=1408, n_layers=24, n_heads=22, n_kv_heads=4,
    d_ff=3840, max_seq_len=4096, vocab_size=40000,
)

MODEL_BASE = ModelConfig(
    d_model=2048, n_layers=22, n_heads=16, n_kv_heads=4,
    d_ff=5504, max_seq_len=2048, vocab_size=32000,
)


@dataclass
class TrainConfig:
    stage: str = "pretrain"  # pretrain | cooldown | sft | dpo

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

    # Precision
    use_bf16: bool = True
    use_gradient_checkpointing: bool = True

    # Checkpointing
    save_every: int = 1000
    eval_every: int = 500
    keep_last_n: int = 3
    checkpoint_dir: str = "checkpoints"
    zip_checkpoints: bool = True

    # Logging
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "seto"

    # Data
    train_data: str = ""
    val_data: str = ""
    tokenizer_path: str = "seto-tokenizer"
    num_workers: int = 4
    max_seq_len: int = 2048

    # Data mixture weights (for pretrain)
    mixture_weights: dict = field(default_factory=lambda: {
        "web_edu": 0.50,
        "books": 0.10,
        "code": 0.15,
        "math_science": 0.10,
        "wiki": 0.10,
        "synthetic": 0.05,
    })

    # Language weights
    lang_weights: dict = field(default_factory=lambda: {
        "en": 0.70,
        "ru": 0.30,
    })

    # SFT config
    sft_data: str = ""
    sft_max_samples: int = 500000
    sft_loss_on_all_tokens: bool = False  # True = standard CE, False = only assistant turns

    # DPO config
    dpo_data: str = ""
    dpo_beta: float = 0.1
    dpo_max_samples: int = 100000

    # Distillation
    teacher_model: str = ""
    distill_alpha: float = 0.5  # weight for distillation loss

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
        return self.effective_batch_size * self.max_seq_len // 2


# Stage presets
STAGE_PRETRAIN = TrainConfig(
    stage="pretrain",
    lr=3e-4, min_lr=3e-5,
    warmup_steps=2000, max_steps=100000,
    batch_size=4, grad_accum_steps=8,
    save_every=1000, eval_every=500,
)

STAGE_COOLDOWN = TrainConfig(
    stage="cooldown",
    lr=1e-4, min_lr=1e-5,
    warmup_steps=100, max_steps=10000,
    batch_size=4, grad_accum_steps=8,
    save_every=500, eval_every=250,
    mixture_weights={
        "web_edu": 0.30,
        "books": 0.15,
        "code": 0.20,
        "math_science": 0.15,
        "wiki": 0.10,
        "synthetic": 0.10,
    },
)

STAGE_SFT = TrainConfig(
    stage="sft",
    lr=5e-5, min_lr=5e-6,
    warmup_steps=200, max_steps=5000,
    batch_size=8, grad_accum_steps=4,
    save_every=500, eval_every=100,
    sft_loss_on_all_tokens=False,
)

STAGE_DPO = TrainConfig(
    stage="dpo",
    lr=5e-6, min_lr=5e-7,
    warmup_steps=100, max_steps=2000,
    batch_size=8, grad_accum_steps=4,
    save_every=500, eval_every=100,
    dpo_beta=0.1,
)
