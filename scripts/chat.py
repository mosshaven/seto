"""Seto inference — chat with your model."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seto.config import ModelConfig
from seto.model import SetoLM
from seto.tokenizer import SetoTokenizer


def load_model(checkpoint_path: str, device: str = "auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
    else:
        config = ModelConfig()

    # Load model
    model = SetoLM(config)

    if os.path.isdir(checkpoint_path):
        weights_path = os.path.join(checkpoint_path, "model.pt")
    else:
        weights_path = checkpoint_path

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config


def load_tokenizer(tokenizer_path: str):
    return SetoTokenizer.from_pretrained(tokenizer_path)


@torch.no_grad()
def generate(
    model: SetoLM,
    tokenizer: SetoTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu",
):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages)
    input_ids = torch.tensor([tokenizer.encode(text, add_bos=True, add_eos=False)], device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-p filtering
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

        # Stop on EOS
        if next_token.item() == tokenizer.eos_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=-1)

    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


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

        text = tokenizer.apply_chat_template(messages)
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

            # Stream output
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            print(token_text, end="", flush=True)

        print()

        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text})


def main():
    parser = argparse.ArgumentParser(description="Chat with Seto")
    parser.add_argument("--model", required=True, help="Path to model checkpoint or directory")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer directory")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--system-prompt", default="Ты Seto — полезный русскоязычный ассистент.", help="System prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading model from {args.model}...")
    model, config = load_model(args.model, device)
    print(f"Model: {config.num_params():,} params")

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Vocab: {len(tokenizer)}")

    chat(model, tokenizer, device, args.system_prompt)


if __name__ == "__main__":
    main()
