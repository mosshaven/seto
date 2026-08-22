#!/usr/bin/env python3
"""Seto inference — chat with your model."""

import argparse
import dataclasses
import json
import os
import sys
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto.config import ModelConfig
from seto.model import SetoLM
from seto.tokenizer import SetoTokenizer


def _zip_member(names: list[str], filename: str, exclude: str = "") -> str:
    matches = [
        name for name in names
        if (name == filename or name.endswith("/" + filename))
        and (not exclude or exclude not in "/" + name)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename} in checkpoint ZIP, found {matches}")
    return matches[0]


def load_model(checkpoint_path: str, device: str = "auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    path = Path(checkpoint_path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            config_member = _zip_member(names, "config.json", "/tokenizer/")
            config_dict = json.loads(archive.read(config_member).decode("utf-8"))
            model_member = _zip_member(names, "model.pt")
            with archive.open(model_member) as model_file:
                state_dict = torch.load(
                    model_file, map_location="cpu", weights_only=True
                )
    else:
        if path.is_dir():
            config_path = path / "config.json"
            weights_path = path / "model.pt"
        else:
            config_path = path.with_name("config.json")
            weights_path = path
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        with config_path.open() as config_file:
            config_dict = json.load(config_file)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    expected_fields = {field.name for field in dataclasses.fields(ModelConfig)}
    missing_fields = sorted(expected_fields - set(config_dict))
    unknown_fields = sorted(set(config_dict) - expected_fields)
    if missing_fields or unknown_fields:
        raise ValueError(
            f"Invalid ModelConfig; missing={missing_fields}, unknown={unknown_fields}"
        )
    config = ModelConfig(**config_dict)
    model = SetoLM(config)
    model.load_state_dict(state_dict)
    del state_dict
    model = model.to(device)
    model.eval()

    return model, config


def load_tokenizer(checkpoint_path: str, tokenizer_path: str | None = None):
    if tokenizer_path:
        tokenizer_override = Path(tokenizer_path)
        if tokenizer_override.suffix != ".zip":
            return SetoTokenizer.from_pretrained(tokenizer_path)
        with zipfile.ZipFile(tokenizer_override) as archive:
            names = archive.namelist()
            tokenizer_member = _zip_member(names, "tokenizer.json")
            tokenizer_json = archive.read(tokenizer_member).decode("utf-8")
            config_matches = [
                name for name in names
                if name == "config.json" or name.endswith("/config.json")
            ]
            config = (
                json.loads(archive.read(config_matches[0]).decode("utf-8"))
                if len(config_matches) == 1
                else None
            )
        return SetoTokenizer.from_serialized(tokenizer_json, config)

    path = Path(checkpoint_path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            tokenizer_member = _zip_member(names, "tokenizer/tokenizer.json")
            config_member = _zip_member(names, "tokenizer/config.json")
            tokenizer_json = archive.read(tokenizer_member).decode("utf-8")
            config = json.loads(archive.read(config_member).decode("utf-8"))
        return SetoTokenizer.from_serialized(tokenizer_json, config)

    model_dir = path if path.is_dir() else path.parent
    candidate = model_dir / "tokenizer"
    if not (candidate / "tokenizer.json").exists():
        raise FileNotFoundError(
            f"Packaged tokenizer not found at {candidate}; pass --tokenizer"
        )
    return SetoTokenizer.from_pretrained(str(candidate))


@torch.no_grad()
def generate(
    model: SetoLM,
    tokenizer: SetoTokenizer,
    messages: list,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu",
):
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([tokenizer.encode(text, add_bos=True, add_eos=False)], device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == tokenizer.eos_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=-1)

    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


@torch.inference_mode()
def chat(model, tokenizer, device, system_prompt=None):
    print("Seto Chat (type 'quit' to exit, 'clear' to reset)")
    print("-" * 50)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "clear":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("[conversation cleared]")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = torch.tensor([tokenizer.encode(text, add_bos=True, add_eos=False)], device=device)

        print("\nSeto: ", end="", flush=True)

        response_tokens = []
        for _ in range(500):
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / 0.7

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_id:
                break

            token_id = next_token.item()
            response_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            print(token_text, end="", flush=True)

        print()

        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text})


def main():
    parser = argparse.ArgumentParser(description="Chat with Seto")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to final ZIP, extracted model directory, or model.pt",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer directory override (auto-loaded from packaged model)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--system-prompt", default="Ты Seto — полезный русскоязычный ассистент.")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model}...")
    model, config = load_model(args.model, device)
    print(f"Model: {config.num_params():,} params")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, args.tokenizer)
    print(f"Vocab: {len(tokenizer)}")

    chat(model, tokenizer, device, args.system_prompt)


if __name__ == "__main__":
    main()
