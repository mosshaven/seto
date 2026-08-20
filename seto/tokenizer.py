"""Seto tokenizer — wrapper around HuggingFace tokenizers."""

import json
import os
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


SPECIAL_TOKENS = {
    "pad": "<|pad|>",
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "unk": "<|unk|>",
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}

CHAT_TEMPLATE = (
    "{system_token}\n{system}\n{user_token}\n{user}\n{assistant_token}\n{assistant}{eos_token}"
)


class SetoTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.special_tokens = SPECIAL_TOKENS
        self._tokenizer: Optional[Tokenizer] = None

    def _build_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS["unk"]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(SPECIAL_TOKENS.values()),
            min_frequency=2,
            show_progress=True,
        )

        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer, trainer

    def train(self, files: List[str], output_dir: str):
        from tokenizers import decoders

        tokenizer, trainer = self._build_tokenizer()
        tokenizer.train(files, trainer)
        tokenizer.enable_padding(pad_id=0, pad_token=SPECIAL_TOKENS["pad"])
        tokenizer.enable_truncation(max_length=2048)

        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "chat_template": CHAT_TEMPLATE,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path: str) -> "SetoTokenizer":
        from tokenizers import decoders

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
            instance.vocab_size = config.get("vocab_size", 32000)
            instance.special_tokens = config.get("special_tokens", SPECIAL_TOKENS)

        return instance

    @property
    def pad_id(self) -> int:
        return self._tokenizer.token_to_id(SPECIAL_TOKENS["pad"])

    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id(SPECIAL_TOKENS["bos"])

    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id(SPECIAL_TOKENS["eos"])

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids
        if add_bos and ids[0] != self.bos_id:
            ids = [self.bos_id] + ids
        if add_eos and ids[-1] != self.eos_id:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: List[dict],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> str:
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            token = self.special_tokens.get(role, f"<|{role}|>")
            parts.append(f"{token}\n{content}")
        parts.append(self.special_tokens["assistant"] + "\n")

        text = "".join(parts)
        if add_bos:
            text = SPECIAL_TOKENS["bos"] + text
        if add_eos:
            text += SPECIAL_TOKENS["eos"]
        return text

    def __len__(self) -> int:
        return self.vocab_size
