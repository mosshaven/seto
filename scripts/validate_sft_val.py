#!/usr/bin/env python3
"""Validate frozen Seto SFT validation data and detect exact train leakage."""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


CATEGORIES = {
    "math",
    "code",
    "explanation",
    "dialogue",
    "safety",
    "tools",
    "ambiguity",
    "toxicity",
    "rewriting",
    "colloquial",
}
LANGUAGES = {"ru", "en"}
ROLES = {"system", "user", "assistant", "tool_call", "tool_result"}
DIFFICULTIES = {"easy", "medium", "hard"}
HUMAN_ORIGINS = {
    "human_authored",
    "public_benchmark",
    "adapted_human",
    "model_draft_human_edited",
}
REVIEW_STATUSES = {"curated", "manually_checked"}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def content_hash(record: dict) -> str:
    return hashlib.sha256(canonical_json(record)).hexdigest()


def normalized_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        content = unicodedata.normalize("NFKC", message["content"]).casefold()
        content = re.sub(r"https?://\S+", "<url>", content)
        content = re.sub(r"\s+", " ", content).strip()
        parts.append(f"{message['role']}:{content}")
    return "\n".join(parts)


def normalized_hash(messages: list[dict]) -> str:
    return hashlib.sha256(normalized_text(messages).encode("utf-8")).hexdigest()


def target_prompt_hashes(messages: list[dict]) -> set[str]:
    return {
        normalized_hash(messages[:index])
        for index, message in enumerate(messages)
        if index > 0 and message.get("role") in {"assistant", "tool_call"}
    }


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(record)
    return records


def validate_messages(messages: object, location: str) -> list[str]:
    errors = []
    if not isinstance(messages, list) or not messages:
        return [f"{location}: messages must be a non-empty list"]

    previous_role = None
    has_target = False
    for index, message in enumerate(messages):
        item = f"{location}:messages[{index}]"
        if not isinstance(message, dict):
            errors.append(f"{item}: expected object")
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ROLES:
            errors.append(f"{item}: invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{item}: content must be non-empty text")
        if role == "system" and index != 0:
            errors.append(f"{item}: system role is only valid first")
        allowed_previous = {
            "system": {None},
            "user": {None, "system", "assistant"},
            "assistant": {"user", "tool_result"},
            "tool_call": {"user", "tool_result"},
            "tool_result": {"tool_call"},
        }
        if role in allowed_previous and previous_role not in allowed_previous[role]:
            errors.append(f"{item}: {role!r} cannot follow {previous_role!r}")
        if role in {"assistant", "tool_call"}:
            has_target = True
        previous_role = role
    if not has_target:
        errors.append(f"{location}: no assistant or tool_call target")
    return errors


def training_hashes(paths: list[Path]) -> tuple[set[str], set[str]]:
    conversation_hashes = set()
    prompt_hashes = set()
    for path in paths:
        for index, record in enumerate(load_jsonl(path), 1):
            messages = record.get("messages", record.get("conversations"))
            if isinstance(messages, list):
                try:
                    conversation_hashes.add(normalized_hash(messages))
                    prompt_hashes.update(target_prompt_hashes(messages))
                except (KeyError, TypeError):
                    print(f"Warning: skipped malformed training row {path}:{index}", file=sys.stderr)
    return conversation_hashes, prompt_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--train", action="append", default=[], type=Path)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--expected-per-category", type=int, default=100)
    parser.add_argument("--expected-per-language-category", type=int, default=50)
    args = parser.parse_args()

    data = load_jsonl(args.data)
    metadata = load_jsonl(args.metadata)
    errors = []
    if len(data) != len(metadata):
        errors.append(f"row count mismatch: data={len(data)}, metadata={len(metadata)}")
    if len(data) != args.expected_count:
        errors.append(f"expected {args.expected_count} rows, found {len(data)}")

    ids = Counter()
    hashes = Counter()
    normalized_hashes = Counter()
    categories = Counter()
    language_categories = Counter()
    train_hashes, train_prompt_hashes = training_hashes(args.train)

    for index, (record, meta) in enumerate(zip(data, metadata), 1):
        location = f"row {index}"
        if set(record) != {"messages"}:
            errors.append(f"{location}: model record must contain only messages")
        messages = record.get("messages")
        message_errors = validate_messages(messages, location)
        errors.extend(message_errors)
        if not isinstance(messages, list):
            continue

        if message_errors:
            continue
        record_hash = content_hash(record)
        norm_hash = normalized_hash(messages)
        hashes[record_hash] += 1
        normalized_hashes[norm_hash] += 1
        if norm_hash in train_hashes:
            errors.append(f"{location}: exact normalized match found in training data")
        leaked_prompts = target_prompt_hashes(messages) & train_prompt_hashes
        if leaked_prompts:
            errors.append(f"{location}: target prompt prefix found in training data")

        required = {
            "id", "line", "language", "category", "difficulty",
            "expected_behavior", "origin_type", "review_status", "source",
            "license", "content_sha256",
        }
        missing = sorted(required - set(meta))
        if missing:
            errors.append(f"{location}: metadata missing {', '.join(missing)}")
            continue
        ids[meta["id"]] += 1
        if meta["line"] != index:
            errors.append(f"{location}: metadata line is {meta['line']!r}")
        if meta["content_sha256"] != record_hash:
            errors.append(f"{location}: content_sha256 mismatch")
        if meta["language"] not in LANGUAGES:
            errors.append(f"{location}: invalid language {meta['language']!r}")
        if meta["category"] not in CATEGORIES:
            errors.append(f"{location}: invalid category {meta['category']!r}")
        if meta["difficulty"] not in DIFFICULTIES:
            errors.append(f"{location}: invalid difficulty {meta['difficulty']!r}")
        if meta["origin_type"] not in HUMAN_ORIGINS:
            errors.append(f"{location}: validation content must have human provenance")
        if meta["review_status"] not in REVIEW_STATUSES:
            errors.append(f"{location}: invalid review_status {meta['review_status']!r}")
        categories[meta["category"]] += 1
        language_categories[(meta["category"], meta["language"])] += 1

    for value, count in ids.items():
        if count > 1:
            errors.append(f"duplicate id {value!r}")
    for value, count in hashes.items():
        if count > 1:
            errors.append(f"duplicate content hash {value}")
    for value, count in normalized_hashes.items():
        if count > 1:
            errors.append(f"duplicate normalized conversation hash {value}")
    for category in sorted(CATEGORIES):
        if categories[category] != args.expected_per_category:
            errors.append(
                f"category {category}: expected {args.expected_per_category}, "
                f"found {categories[category]}"
            )
        for language in sorted(LANGUAGES):
            count = language_categories[(category, language)]
            if count != args.expected_per_language_category:
                errors.append(
                    f"category {category}/{language}: expected "
                    f"{args.expected_per_language_category}, found {count}"
                )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated {len(data)} frozen validation conversations")


if __name__ == "__main__":
    main()
