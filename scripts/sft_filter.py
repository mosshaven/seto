#!/usr/bin/env python3
"""SFT data filter — cleans conversation datasets for SFT training.

Reads JSONL with {"messages": [...]} format.
Outputs filtered JSONL + stats.

Usage:
    python scripts/sft_filter.py --input raw.jsonl --output filtered.jsonl \
        --roleplay-output roleplay.jsonl --stats
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter


# === Filter patterns ===

# Phrases that indicate model boilerplate. Reject only when little useful text
# remains after removing them.
STERILE_PHRASES = [
    # English
    "as an ai language model",
    "as an ai, i",
    "i'm just an ai",
    "i'm an ai assistant",
    "i don't have personal opinions",
    "as a language model, i",
    "i'm programmed to",
    "i'm designed to be helpful",
    "i aim to be helpful",
    "i strive to provide",
    "please note that i",
    "i want to clarify that",
    "to clarify, i",
    "i should mention that",
    # Russian
    "я языковая модель",
    "я — языковая модель",
    "я просто языковая модель",
    "я искусственный интеллект",
    "я не имею личного мнения",
]

# Jailbreak / roleplay patterns
JAILBREAK_PATTERNS = [
    r"^(ignore|forget|disregard)\s+(all\s+)?(previous|prior|earlier|above)",
    r"^(do|act|pretend)\s+(as\s+)?if\s+you\s+(are|were|have)",
    r"^you\s+are\s+now\s+",
    r"^from\s+now\s+on\s+you\s+are",
    r"^jailbreak",
    r"^dan\s+mode",
    r"^developer\s+mode",
    r"^unlock\s+",
    r"^\*?you\s+are\s+a\s+",
    r"^(system|assistant)\s*:\s*",
    r"^(EOF|---)\s*$",
]

# Russian jailbreak
JAILBREAK_PATTERNS_RU = [
    r"^(забудь|забыть|забудьте)\s+(всё|все)\s+(предыдущее|предыдущие)",
    r"^(притворись|притворяйся)\s+что\s+ты",
    r"^ты\s+теперь\s+",
    r"^с\s+этого\s+момента\s+ты",
]


def is_too_short(messages: list, min_assistant_len: int = 10) -> bool:
    """Check if conversation is too short to be useful."""
    target_messages = [
        message for message in messages
        if isinstance(message, dict) and message.get("role") in {"assistant", "tool_call"}
    ]
    if not target_messages:
        return True

    has_substantial = any(
        len(m.get("content", "").strip()) >= min_assistant_len
        for m in target_messages
        if isinstance(m.get("content"), str)
    )
    return not has_substantial


def is_too_long(messages: list, max_turns: int = 20) -> bool:
    """Check if conversation has too many turns."""
    return len(messages) > max_turns


def has_empty_assistant(messages: list) -> bool:
    """Check if any trainable target message is empty."""
    for m in messages:
        if m.get("role") in {"assistant", "tool_call"}:
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                return True
    return False


def has_sterile_response(messages: list, min_useful_chars: int = 80) -> bool:
    """Reject model boilerplate only when it lacks a useful continuation."""
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "").strip().lower()
            matched = [phrase for phrase in STERILE_PHRASES if phrase in content]
            if not matched:
                continue

            useful = content
            for phrase in matched:
                useful = useful.replace(phrase, " ")
            useful = re.sub(r"[^\w\s]+", " ", useful, flags=re.UNICODE)
            useful = re.sub(r"\s+", " ", useful).strip()
            if len(useful) < min_useful_chars:
                return True
    return False


def has_jailbreak(messages: list) -> bool:
    """Check for jailbreak attempts in user messages."""
    all_patterns = JAILBREAK_PATTERNS + JAILBREAK_PATTERNS_RU
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "").strip().lower()
            for pattern in all_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
    return False


def has_duplicate_responses(messages: list) -> bool:
    """Check if assistant gives identical responses to different queries."""
    assistant_contents = []
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "").strip()
            if content:
                assistant_contents.append(content)

    if len(assistant_contents) < 2:
        return False

    # Check for exact duplicates
    seen = set()
    for c in assistant_contents:
        if c in seen:
            return True
        seen.add(c)

    return False


def has_system_prompt_too_long(messages: list, max_len: int = 500) -> bool:
    """Check if system prompt is too long."""
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if len(content) > max_len:
                return True
    return False


def has_bad_formatting(messages: list) -> bool:
    """Check for bad formatting patterns."""
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "")

            # Excessive newlines
            if "\n\n\n\n" in content:
                return True

            # Excessive repetition
            if len(content) > 100:
                words = content.split()
                if len(words) > 20:
                    # Check if same word appears too often
                    word_counts = Counter(w.lower() for w in words)
                    most_common_count = word_counts.most_common(1)[0][1]
                    if most_common_count > len(words) * 0.3:
                        return True

    return False


def is_roleplay_content(messages: list) -> bool:
    """Check for roleplay/character content."""
    rp_patterns = [
        r"^\*[^*]+\*$",  # Actions in asterisks
        r"^\"[^\"]+\"\s*\(",  # "dialog" (action)
        r"^(narrator|system)\s*:\s*",
    ]

    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "").strip()
            for pattern in rp_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    return True

    return False


def filter_conversation(
    messages: list,
    max_turns: int = 20,
    max_system_len: int = 500,
    min_assistant_len: int = 10,
) -> tuple[bool, list[str]]:
    """Filter a single conversation. Returns (pass, reasons)."""
    reasons = []

    if not isinstance(messages, list) or not messages or any(
        not isinstance(message, dict)
        or message.get("role") not in {"system", "user", "assistant", "tool_call", "tool_result"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        return False, ["malformed_messages"]

    if is_too_short(messages, min_assistant_len):
        reasons.append("too_short")
    if is_too_long(messages, max_turns):
        reasons.append("too_long")
    if has_empty_assistant(messages):
        reasons.append("empty_assistant")
    if has_sterile_response(messages):
        reasons.append("sterile")
    if has_jailbreak(messages):
        reasons.append("jailbreak")
    if has_duplicate_responses(messages):
        reasons.append("duplicate")
    if has_system_prompt_too_long(messages, max_system_len):
        reasons.append("system_too_long")
    if has_bad_formatting(messages):
        reasons.append("bad_formatting")
    return len(reasons) == 0, reasons


def main():
    parser = argparse.ArgumentParser(description="Filter SFT conversations")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--roleplay-output",
        required=True,
        help="JSONL for otherwise valid roleplay conversations",
    )
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    parser.add_argument("--max-turns", type=int, default=20, help="Max conversation turns")
    parser.add_argument("--max-system-len", type=int, default=500, help="Max system prompt length")
    parser.add_argument("--min-assistant-len", type=int, default=10, help="Min assistant response length")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    roleplay_path = Path(args.roleplay_output)

    if output_path.resolve() == roleplay_path.resolve():
        parser.error("--output and --roleplay-output must differ")

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Read all JSONL files
    conversations = []
    files = list(input_path.parent.glob(input_path.name)) if "*" in str(input_path) else [input_path]

    for f in files:
        try:
            with open(f) as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        messages = obj.get("messages", obj.get("conversations", []))
                        if messages:
                            conversations.append(messages)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    print(f"Loaded {len(conversations)} conversations from {len(files)} files")

    # Filter
    passed = []
    roleplay = []
    rejected = Counter()

    for msgs in conversations:
        ok, reasons = filter_conversation(
            msgs,
            max_turns=args.max_turns,
            max_system_len=args.max_system_len,
            min_assistant_len=args.min_assistant_len,
        )
        if ok:
            if is_roleplay_content(msgs):
                roleplay.append(msgs)
            else:
                passed.append(msgs)
        else:
            for r in reasons:
                rejected[r] += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for msgs in passed:
            fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    roleplay_path.parent.mkdir(parents=True, exist_ok=True)
    with open(roleplay_path, "w") as fh:
        for msgs in roleplay:
            fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    print(f"Passed: {len(passed)} / {len(conversations)} ({len(passed)/max(len(conversations),1)*100:.1f}%)")
    print(f"Roleplay separated: {len(roleplay)}")
    print(f"Rejected: {len(conversations) - len(passed) - len(roleplay)}")

    if args.stats or rejected:
        print("\nRejection reasons:")
        for reason, count in rejected.most_common():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
