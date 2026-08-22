#!/usr/bin/env python3
"""Build a deterministic weighted SFT mix from canonical JSONL sources."""

import argparse
import glob
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path


TARGET_ROLES = {"assistant", "tool_call"}
VALID_ROLES = {"system", "user", "assistant", "tool_call", "tool_result"}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        content = unicodedata.normalize("NFKC", message["content"]).casefold()
        content = re.sub(r"https?://\S+", "<url>", content)
        content = re.sub(r"\s+", " ", content).strip()
        parts.append(f"{message['role']}:{content}")
    return "\n".join(parts)


def normalized_hash(messages: list[dict]) -> str:
    return sha256(normalized_text(messages).encode("utf-8"))


def target_prompt_hashes(messages: list[dict]) -> set[str]:
    return {
        normalized_hash(messages[:index])
        for index, message in enumerate(messages)
        if index > 0 and message["role"] in TARGET_ROLES
    }


def immediate_request_hashes(messages: list[dict]) -> set[str]:
    return {
        normalized_hash([messages[index - 1]])
        for index, message in enumerate(messages)
        if index > 0
        and message["role"] in TARGET_ROLES
        and messages[index - 1]["role"] in {"user", "tool_result"}
    }


def validate_messages(messages: object) -> str | None:
    if not isinstance(messages, list) or not messages:
        return "missing_messages"
    targets = 0
    for message in messages:
        if not isinstance(message, dict):
            return "malformed_message"
        if message.get("role") not in VALID_ROLES:
            return "invalid_role"
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            return "empty_content"
        if message["role"] in TARGET_ROLES:
            targets += 1
    if targets == 0:
        return "no_trainable_target"
    return None


def target_count(messages: list[dict]) -> int:
    return sum(message["role"] in TARGET_ROLES for message in messages)


def stable_key(seed: int, namespace: str, digest: str) -> str:
    return sha256(f"{seed}:{namespace}:{digest}".encode("utf-8"))


def display_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def expand_paths(patterns: list[str], base_dir: Path) -> list[Path]:
    paths = {
        Path(match).resolve()
        for pattern in patterns
        for match in glob.glob(
            str(Path(pattern) if Path(pattern).is_absolute() else base_dir / pattern),
            recursive=True,
        )
    }
    return sorted(path for path in paths if path.is_file())


def validation_hashes(patterns: list[str], base_dir: Path) -> set[str]:
    hashes = set()
    for pattern in patterns:
        paths = expand_paths([pattern], base_dir)
        if not paths:
            raise ValueError(f"validation pattern {pattern!r} matched no files")
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                    messages = record.get("messages") if isinstance(record, dict) else None
                    reason = validate_messages(messages)
                    if reason:
                        raise ValueError(f"{path}:{line_number}: {reason}")
                    hashes.add(normalized_hash(messages))
                    hashes.update(target_prompt_hashes(messages))
                    hashes.update(immediate_request_hashes(messages))
    return hashes


def load_source(
    source: dict,
    seed: int,
    blocked_hashes: set[str],
    base_dir: Path,
) -> tuple[list[dict], Counter]:
    records = []
    stats = Counter()
    seen_hashes = set()
    paths = expand_paths(source["paths"], base_dir)
    if not paths:
        raise ValueError(f"source {source['name']!r} matched no files")

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stats["input_records"] += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid_json"] += 1
                    continue
                messages = raw.get("messages") if isinstance(raw, dict) else None
                reason = validate_messages(messages)
                if reason:
                    stats[reason] += 1
                    continue
                digest = normalized_hash(messages)
                leakage_keys = target_prompt_hashes(messages) | immediate_request_hashes(messages)
                if digest in blocked_hashes or leakage_keys & blocked_hashes:
                    stats["validation_leakage"] += 1
                    continue
                if digest in seen_hashes:
                    stats["duplicate"] += 1
                    continue
                seen_hashes.add(digest)
                record = {"messages": messages}
                records.append(
                    {
                        "record": record,
                        "digest": digest,
                        "targets": target_count(messages),
                        "source_file": display_path(path, base_dir),
                        "source_line": line_number,
                    }
                )
    records.sort(key=lambda item: stable_key(seed, source["name"], item["digest"]))
    return records, stats


def allocate_targets(total: int, sources: list[dict]) -> dict[str, int]:
    weight_sum = sum(source["weight"] for source in sources)
    raw = {source["name"]: total * source["weight"] / weight_sum for source in sources}
    allocation = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(allocation.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - allocation[name]), name))
    for name in order[:remainder]:
        allocation[name] += 1
    return allocation


def select_to_quota(records: list[dict], quota: int) -> tuple[list[dict], list[dict]]:
    selected = []
    leftovers = []
    used = 0
    for item in records:
        if used + item["targets"] <= quota:
            selected.append(item)
            used += item["targets"]
        else:
            leftovers.append(item)
    return selected, leftovers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_dir = (args.config.parent / config.get("base_dir", ".")).resolve()
    seed = int(config.get("seed", 42))
    requested_targets = int(config["target_examples"])
    sources = config["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    if any(
        not isinstance(source, dict)
        or not isinstance(source.get("name"), str)
        or not source["name"]
        or not isinstance(source.get("paths"), list)
        or not source["paths"]
        or not all(isinstance(path, str) and path for path in source["paths"])
        for source in sources
    ):
        raise ValueError("each source requires a name and non-empty paths list")
    names = [source["name"] for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("source names must be unique")
    if requested_targets <= 0:
        raise ValueError("target_examples must be positive")
    if any(
        isinstance(source.get("weight"), bool)
        or not isinstance(source.get("weight"), (int, float))
        or not math.isfinite(source["weight"])
        or source["weight"] <= 0
        for source in sources
    ):
        raise ValueError("source weights must be positive")

    blocked = validation_hashes(config.get("validation_paths", []), base_dir)
    pools = {}
    source_stats = {}
    for source in sources:
        pools[source["name"]], stats = load_source(source, seed, blocked, base_dir)
        source_stats[source["name"]] = stats

    owners = {}
    candidates = {}
    for name, records in pools.items():
        for item in records:
            candidates.setdefault(item["digest"], []).append(name)
    for digest, source_names in candidates.items():
        owners[digest] = min(
            source_names,
            key=lambda name: stable_key(seed, f"owner:{digest}", name),
        )
    for name, records in pools.items():
        source_stats[name]["cross_source_duplicate"] += sum(
            owners[item["digest"]] != name for item in records
        )
        pools[name] = [item for item in records if owners[item["digest"]] == name]
        source_stats[name]["eligible_records"] = len(pools[name])
        source_stats[name]["eligible_targets"] = sum(
            item["targets"] for item in pools[name]
        )

    quotas = allocate_targets(requested_targets, sources)
    selected = {}
    leftovers = {}
    for source in sources:
        name = source["name"]
        selected[name], leftovers[name] = select_to_quota(pools[name], quotas[name])

    # Redistribute unavailable quota without duplicating records.
    while True:
        used = sum(item["targets"] for items in selected.values() for item in items)
        deficit = requested_targets - used
        available = [source for source in sources if leftovers[source["name"]]]
        if deficit <= 0 or not available:
            break
        extra_quotas = allocate_targets(deficit, available)
        progress = 0
        for source in available:
            name = source["name"]
            extra, leftovers[name] = select_to_quota(leftovers[name], extra_quotas[name])
            selected[name].extend(extra)
            progress += sum(item["targets"] for item in extra)
        if progress == 0:
            break

    mixed = [
        {**item, "source": name}
        for name, items in selected.items()
        for item in items
    ]
    mixed.sort(key=lambda item: stable_key(seed, "mixed", item["digest"]))

    output_lines = []
    metadata_lines = []
    selected_counts = Counter()
    selected_targets = Counter()
    for line_number, item in enumerate(mixed, 1):
        record_bytes = canonical_json(item["record"])
        output_lines.append(record_bytes)
        selected_counts[item["source"]] += 1
        selected_targets[item["source"]] += item["targets"]
        metadata_lines.append(
            canonical_json(
                {
                    "line": line_number,
                    "source": item["source"],
                    "source_file": item["source_file"],
                    "source_line": item["source_line"],
                    "trainable_targets": item["targets"],
                    "content_sha256": sha256(record_bytes),
                    "normalized_sha256": item["digest"],
                }
            )
        )

    output = b"".join(line + b"\n" for line in output_lines)
    metadata = b"".join(line + b"\n" for line in metadata_lines)
    total_targets = sum(selected_targets.values())
    manifest = {
        "version": 1,
        "config": config,
        "requested_targets": requested_targets,
        "selected_records": len(mixed),
        "selected_targets": total_targets,
        "target_deficit": requested_targets - total_targets,
        "validation_hashes": len(blocked),
        "sources": {
            source["name"]: {
                "requested_weight": source["weight"],
                "requested_targets": quotas[source["name"]],
                "selected_records": selected_counts[source["name"]],
                "selected_targets": selected_targets[source["name"]],
                "actual_target_weight": (
                    selected_targets[source["name"]] / total_targets if total_targets else 0
                ),
                "filter_stats": dict(source_stats[source["name"]]),
            }
            for source in sources
        },
        "files": {
            "output_sha256": sha256(output),
            "metadata_sha256": sha256(metadata),
        },
    }

    paths = (args.output, args.metadata_output, args.manifest_output)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("output paths must differ")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.metadata_output.write_bytes(metadata)
    args.manifest_output.write_bytes(canonical_json(manifest) + b"\n")
    print(f"Wrote {len(mixed)} conversations / {total_targets} trainable targets")


if __name__ == "__main__":
    main()
