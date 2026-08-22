#!/usr/bin/env python3
"""Stream and normalize public datasets for the first Seto SFT mix."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from sft_filter import filter_conversation, is_roleplay_content


SOURCES = {
    "ultrachat": {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "config": None,
        "split": "train_sft",
        "license": "MIT",
    },
    "wildchat": {
        "dataset": "allenai/WildChat-1M",
        "revision": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
        "config": None,
        "split": "train",
        "license": "ODC-BY-1.0",
    },
    "russian": {
        "dataset": "attn-signs/russian-easy-instructions",
        "revision": "77a180e009a0c9994b58264caa0353531764f1e8",
        "config": None,
        "split": "train",
        "license": "Apache-2.0",
    },
    "tools": {
        "dataset": "NousResearch/hermes-function-calling-v1",
        "revision": "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
        "config": "func_calling",
        "split": "train",
        "license": "Apache-2.0",
    },
}

ROLE_MAP = {
    "human": "user",
    "prompter": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool_result",
}

CASUAL_BLOCKLIST = (
    "write an essay",
    "write a story",
    "write a detailed",
    "professional tone",
    "generate a report",
    "create an article",
    "act as ",
    "ignore previous",
)


def digest(messages: list[dict]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def target_count(messages: list[dict]) -> int:
    return sum(message["role"] in {"assistant", "tool_call"} for message in messages)


def normalize_messages(raw_messages) -> list[dict] | None:
    if not isinstance(raw_messages, list):
        return None
    messages = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            return None
        role = ROLE_MAP.get(raw.get("role", raw.get("from")))
        content = raw.get("content", raw.get("value"))
        if role not in {"system", "user", "assistant"}:
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        messages.append({"role": role, "content": content.strip()})
    if messages and messages[0]["role"] == "system":
        body = messages[1:]
    else:
        body = messages
    if not body or body[0]["role"] != "user":
        return None
    if any(
        message["role"] != ("user" if index % 2 == 0 else "assistant")
        for index, message in enumerate(body)
    ):
        return None
    if body[-1]["role"] != "assistant":
        return None
    return messages


def general_quality(messages: list[dict], max_turns: int = 12) -> bool:
    if sum(len(message["content"]) for message in messages) > 8000:
        return False
    passed, _ = filter_conversation(
        messages,
        max_turns=max_turns,
        max_system_len=1000,
        min_assistant_len=10,
    )
    return passed and not is_roleplay_content(messages)


def stream_source(name: str, seed: int):
    source = SOURCES[name]
    kwargs = {
        "path": source["dataset"],
        "split": source["split"],
        "revision": source["revision"],
        "streaming": True,
    }
    if source["config"]:
        kwargs["name"] = source["config"]
    dataset = load_dataset(**kwargs)
    return dataset.shuffle(seed=seed, buffer_size=10000)


def write_until(path: Path, records, target_quota: int) -> dict:
    seen = set()
    stats = Counter()
    targets = 0
    with path.open("w", encoding="utf-8") as output:
        for messages in records:
            stats["seen"] += 1
            if messages is None:
                stats["rejected"] += 1
                continue
            key = digest(messages)
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            output.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count = target_count(messages)
            targets += count
            stats["records"] += 1
            stats["targets"] += count
            if targets >= target_quota:
                break
    if targets < target_quota:
        raise RuntimeError(f"{path.name}: only {targets}/{target_quota} targets available")
    return dict(stats)


def ultrachat_records(seed: int):
    for row in stream_source("ultrachat", seed):
        messages = normalize_messages(row.get("messages"))
        yield messages if messages and general_quality(messages) else None


def wildchat_records(seed: int, category: str):
    for row in stream_source("wildchat", seed):
        if row.get("toxic") is True:
            yield None
            continue
        raw_messages = row.get("conversation")
        if isinstance(raw_messages, list) and any(
            message.get("toxic") is True for message in raw_messages
            if isinstance(message, dict)
        ):
            yield None
            continue
        messages = normalize_messages(raw_messages)
        if not messages or not general_quality(messages, max_turns=10):
            yield None
            continue
        assistants = target_count(messages)
        first_user = next(
            message["content"] for message in messages if message["role"] == "user"
        )
        if category == "dialogue":
            yield messages if assistants >= 2 else None
            continue
        blocked = any(term in first_user.lower() for term in CASUAL_BLOCKLIST)
        is_casual = (
            assistants <= 2
            and len(first_user) <= 300
            and sum(len(message["content"]) for message in messages) <= 3000
            and not blocked
        )
        yield messages if is_casual else None


def russian_records(seed: int):
    for row in stream_source("russian", seed):
        messages = normalize_messages(row.get("conversation"))
        if messages is None:
            instruction = row.get("instruction")
            answer = row.get("answer")
            if isinstance(instruction, str) and isinstance(answer, str):
                messages = [
                    {"role": "user", "content": instruction.strip()},
                    {"role": "assistant", "content": answer.strip()},
                ]
        yield messages if messages and general_quality(messages) else None


def extract_tagged_json(text: str, tag: str):
    blocks = re.findall(
        rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE
    )
    if not blocks:
        return None
    values = []
    for block in blocks:
        try:
            values.append(json.loads(block))
        except json.JSONDecodeError:
            return None
    value = values[0] if len(values) == 1 else values
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tool_messages(row: dict) -> list[dict] | None:
    try:
        tools = json.loads(row.get("tools", ""))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(tools, list) or not tools:
        return None
    messages = [
        {
            "role": "system",
            "content": "Available tools:\n" + json.dumps(
                tools, ensure_ascii=False, separators=(",", ":")
            ),
        }
    ]
    for raw in row.get("conversations", []):
        sender = raw.get("from")
        value = raw.get("value")
        if not isinstance(value, str) or not value.strip():
            return None
        if sender == "system":
            continue
        if sender == "human":
            messages.append({"role": "user", "content": value.strip()})
        elif sender == "tool":
            content = extract_tagged_json(value, "tool_response")
            if content is None:
                return None
            messages.append({"role": "tool_result", "content": content})
        elif sender == "gpt":
            if "<tool_call>" in value:
                content = extract_tagged_json(value, "tool_call")
                if content is None:
                    return None
                messages.append({"role": "tool_call", "content": content})
            else:
                messages.append({"role": "assistant", "content": value.strip()})
        else:
            return None
    if not any(message["role"] == "user" for message in messages):
        return None
    if not any(message["role"] == "tool_call" for message in messages):
        return None
    if sum(len(message["content"]) for message in messages) > 8000:
        return None
    return messages


def tools_records(seed: int):
    for row in stream_source("tools", seed):
        yield tool_messages(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ultrachat-targets", type=int, default=22000)
    parser.add_argument("--dialogue-targets", type=int, default=6500)
    parser.add_argument("--casual-targets", type=int, default=8500)
    parser.add_argument("--russian-targets", type=int, default=6000)
    parser.add_argument("--tools-targets", type=int, default=6000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "ultrachat": (ultrachat_records(args.seed), args.ultrachat_targets),
        "wildchat-dialogue": (
            wildchat_records(args.seed + 1, "dialogue"), args.dialogue_targets
        ),
        "wildchat-casual": (
            wildchat_records(args.seed + 2, "casual"), args.casual_targets
        ),
        "russian": (russian_records(args.seed + 3), args.russian_targets),
        "tools": (tools_records(args.seed + 4), args.tools_targets),
    }
    stats = {}
    for name, (records, quota) in jobs.items():
        print(f"Preparing {name}: target={quota}", flush=True)
        stats[name] = write_until(args.output_dir / f"{name}.jsonl", records, quota)
        print(f"  {stats[name]}", flush=True)

    manifest = {
        "seed": args.seed,
        "sources": SOURCES,
        "stats": stats,
    }
    (args.output_dir / "sources.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
