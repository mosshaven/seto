"""Seto tokenizer — train from scratch on multilingual data (RU/UK/EN)."""

import json
import os
from typing import List, Optional

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders


SPECIAL_TOKENS = {
    "pad": "<|pad|>",
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "unk": "<|unk|>",
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool_call": "<|tool_call|>",
    "tool_result": "<|tool_result|>",
}


class SetoTokenizer:
    def __init__(self, vocab_size: int = 48000):
        self.vocab_size = vocab_size
        self.special_tokens = SPECIAL_TOKENS
        self._tokenizer: Optional[Tokenizer] = None

    def _build_tokenizer(self) -> tuple:
        tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS["unk"]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(SPECIAL_TOKENS.values()),
            min_frequency=2,
            show_progress=True,
        )

        return tokenizer, trainer

    def train(self, files: List[str], output_dir: str):
        tokenizer, trainer = self._build_tokenizer()
        tokenizer.train(files, trainer)
        tokenizer.enable_padding(pad_id=0, pad_token=SPECIAL_TOKENS["pad"])
        tokenizer.enable_truncation(max_length=8192)

        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path: str) -> "SetoTokenizer":
        tokenizer_file = os.path.join(path, "tokenizer.json")
        config_file = os.path.join(path, "config.json")

        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

        instance = cls()
        instance._tokenizer = Tokenizer.from_file(tokenizer_file)
        instance._tokenizer.decoder = decoders.ByteLevel()

        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
            instance.vocab_size = config.get("vocab_size", 48000)
            instance.special_tokens = {
                **SPECIAL_TOKENS,
                **config.get("special_tokens", {}),
            }

        return instance

    @classmethod
    def from_serialized(
        cls,
        tokenizer_json: str,
        config: Optional[dict] = None,
    ) -> "SetoTokenizer":
        """Load tokenizer from serialized JSON, for packaged checkpoints."""
        instance = cls()
        instance._tokenizer = Tokenizer.from_str(tokenizer_json)
        instance._tokenizer.decoder = decoders.ByteLevel()
        if config:
            instance.vocab_size = config.get("vocab_size", 48000)
            instance.special_tokens = {
                **SPECIAL_TOKENS,
                **config.get("special_tokens", {}),
            }
        instance.vocab_size = instance._tokenizer.get_vocab_size(
            with_added_tokens=True
        )
        return instance

    @property
    def pad_id(self) -> int:
        return self._tokenizer.token_to_id(self.special_tokens["pad"])

    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id(self.special_tokens["bos"])

    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id(self.special_tokens["eos"])

    @property
    def tool_call_id(self) -> int:
        return self._tokenizer.token_to_id(self.special_tokens["tool_call"])

    @property
    def tool_result_id(self) -> int:
        return self._tokenizer.token_to_id(self.special_tokens["tool_result"])

    def add_special_tokens(self, tokens: List[str]) -> int:
        """Add new special tokens to existing tokenizer. Returns number added."""
        added = self._tokenizer.add_special_tokens(tokens)
        self.vocab_size = self._tokenizer.get_vocab_size(with_added_tokens=True)
        return added

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids
        if add_bos and (not ids or ids[0] != self.bos_id):
            ids = [self.bos_id] + ids
        if add_eos and (not ids or ids[-1] != self.eos_id):
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: List[dict], add_generation_prompt: bool = True) -> str:
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            token = self.special_tokens.get(role, f"<|{role}|>")
            parts.append(f"{token}\n{content}")
        if add_generation_prompt:
            parts.append(self.special_tokens["assistant"] + "\n")
        text = self.special_tokens["bos"] + "".join(parts)
        if not add_generation_prompt:
            text += self.special_tokens["eos"]
        return text

    def __len__(self) -> int:
        if self._tokenizer is None:
            return self.vocab_size
        return self._tokenizer.get_vocab_size(with_added_tokens=True)
