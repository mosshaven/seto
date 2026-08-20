"""Seto — tiny language model for mobile deployment."""

from .config import ModelConfig, TrainConfig, MODEL_TINY, MODEL_SMALL, MODEL_BASE
from .config import STAGE_PRETRAIN, STAGE_COOLDOWN, STAGE_SFT, STAGE_DPO
from .model import SetoLM
from .tokenizer import SetoTokenizer
from .checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from .data import ShardDataset, StreamingShardDataset, pack_tokens_to_shards, find_shards, create_dataloader
from .trainer import SetoTrainer
from .sft import SFTTrainer
from .dpo import DPOTrainer
from .data_filter import filter_text, detect_language, compute_quality_score

__version__ = "0.1.0"
__all__ = [
    "ModelConfig", "TrainConfig",
    "MODEL_TINY", "MODEL_SMALL", "MODEL_BASE",
    "STAGE_PRETRAIN", "STAGE_COOLDOWN", "STAGE_SFT", "STAGE_DPO",
    "SetoLM", "SetoTokenizer", "SetoTrainer", "SFTTrainer", "DPOTrainer",
    "save_checkpoint", "load_checkpoint", "get_latest_checkpoint",
    "ShardDataset", "StreamingShardDataset", "pack_tokens_to_shards",
    "find_shards", "create_dataloader",
    "filter_text", "detect_language", "compute_quality_score",
]
