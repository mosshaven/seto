#!/usr/bin/env python3
"""Convert OpenAssistant/oasst1 trees to deterministic SFT JSONL."""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SOURCE_DATASET = "OpenAssistant/oasst1"
DEFAULT_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
CONVERTER_VERSION = 1


@dataclass(frozen=True)
class Node:
    message_id: str
    parent_id: str | None
    tree_id: str
    text: Any
    role: Any
    lang: Any
    deleted: Any
    rank: Any
    review_count: Any
    review_result: Any
    synthetic: Any


@dataclass(frozen=True)
class Candidate:
    tree_id: str
    leaf_id: str
    language: str
    nodes: tuple[Node, ...]
    ranks: tuple[int | None, ...]


def canonical_json(value: Any) -> bytes:
    """Serialize values identically across input ordering and locales."""
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_languages(value: str) -> tuple[str, ...]:
    languages = tuple(sorted({item.strip().lower() for item in value.split(",") if item.strip()}))
    if not languages:
        raise argparse.ArgumentTypeError("at least one language is required")
    return languages


def parse_node(row: Any) -> tuple[Node | None, str | None]:
    if not isinstance(row, dict):
        return None, "malformed_record"
    message_id = row.get("message_id")
    tree_id = row.get("message_tree_id")
    parent_id = row.get("parent_id")
    if not isinstance(message_id, str) or not message_id:
        return None, "invalid_message_id"
    if not isinstance(tree_id, str) or not tree_id:
        return None, "invalid_tree_id"
    if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
        return None, "invalid_parent_id"
    return Node(
        message_id=message_id,
        parent_id=parent_id,
        tree_id=tree_id,
        text=row.get("text"),
        role=row.get("role"),
        lang=row.get("lang"),
        deleted=row.get("deleted"),
        rank=row.get("rank"),
        review_count=row.get("review_count"),
        review_result=row.get("review_result"),
        synthetic=row.get("synthetic"),
    ), None


def candidate_rejection(
    path: tuple[Node, ...],
    languages: set[str],
    max_turns: int | None,
    max_chars: int | None,
    max_rank: int | None,
    min_reviews: int,
) -> str | None:
    if any(type(node.deleted) is not bool for node in path):
        return "malformed_deleted"
    if any(node.deleted for node in path):
        return "deleted"
    if any(type(node.synthetic) is not bool for node in path):
        return "malformed_synthetic"
    if any(node.synthetic for node in path):
        return "synthetic"
    if any(not isinstance(node.text, str) for node in path):
        return "malformed_text"
    if any(not node.text.strip() for node in path):
        return "empty_text"
    if any(node.role not in {"prompter", "assistant"} for node in path):
        return "malformed_role"
    expected = ("prompter", "assistant")
    if any(node.role != expected[index % 2] for index, node in enumerate(path)):
        return "invalid_role_alternation"
    if path[-1].role != "assistant":
        return "non_assistant_leaf"
    if any(not isinstance(node.lang, str) or not node.lang.strip() for node in path):
        return "malformed_language"
    path_languages = {node.lang.strip().lower() for node in path}
    if len(path_languages) != 1:
        return "mixed_language"
    if not path_languages.issubset(languages):
        return "language_not_selected"
    if max_turns is not None and len(path) > max_turns:
        return "too_many_turns"
    if max_chars is not None and sum(len(node.text) for node in path) > max_chars:
        return "too_many_chars"

    for node in path:
        if type(node.review_count) is not int or node.review_count < 0:
            return "malformed_review_count"
        if type(node.review_result) is not bool:
            return "malformed_review_result"
        if node.role == "assistant" and not node.review_result:
            return "assistant_review_failed"
        if node.role == "assistant" and node.review_count < min_reviews:
            return "insufficient_reviews"
        if node.rank is None:
            continue
        if type(node.rank) is not int or node.rank < 0:
            return "malformed_rank"
        if node.role == "assistant" and max_rank is not None and node.rank > max_rank:
            return "rank_above_max"
    return None


def selection_key(candidate: Candidate) -> tuple[Any, ...]:
    """Prefer complete ranks, lower mean rank, depth, then leaf ID."""
    known = tuple(rank for rank in candidate.ranks if rank is not None)
    complete = len(known) == len(candidate.ranks)
    mean = Fraction(sum(known), len(known)) if known else Fraction(sys.maxsize, 1)
    return (not complete, mean, -len(candidate.nodes), candidate.leaf_id)


def validate_tree(
    nodes: list[Node],
    global_nodes: dict[str, Node],
    duplicate_ids: set[str],
    cross_tree_ids: set[str],
) -> tuple[Node | None, dict[str, list[Node]], str | None]:
    if any(node.message_id in duplicate_ids for node in nodes):
        return None, {}, "duplicate_message_id"
    if nodes[0].tree_id in cross_tree_ids:
        return None, {}, "cross_tree_parent"
    roots = [node for node in nodes if node.parent_id is None]
    if len(roots) != 1:
        return None, {}, "invalid_root_count"

    children: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        if node.parent_id is None:
            continue
        if node.parent_id == node.message_id:
            return None, {}, "cycle"
        parent = global_nodes.get(node.parent_id)
        if parent is None:
            return None, {}, "missing_parent"
        if parent.tree_id != node.tree_id:
            return None, {}, "cross_tree_parent"
        children[node.parent_id].append(node)
    for siblings in children.values():
        siblings.sort(key=lambda node: node.message_id)

    visited: set[str] = set()
    active: set[str] = set()

    def visit(node: Node) -> bool:
        if node.message_id in active:
            return False
        if node.message_id in visited:
            return True
        active.add(node.message_id)
        for child in children.get(node.message_id, ()):
            if not visit(child):
                return False
        active.remove(node.message_id)
        visited.add(node.message_id)
        return True

    if not visit(roots[0]):
        return None, {}, "cycle"
    if len(visited) != len(nodes):
        return None, {}, "disconnected"
    return roots[0], dict(children), None


def leaf_paths(root: Node, children: dict[str, list[Node]]) -> Iterable[tuple[Node, ...]]:
    stack: list[tuple[Node, tuple[Node, ...]]] = [(root, ())]
    while stack:
        node, prefix = stack.pop()
        path = prefix + (node,)
        descendants = children.get(node.message_id, ())
        if not descendants:
            yield path
            continue
        for child in reversed(descendants):
            stack.append((child, path))


def convert_rows(
    rows: Iterable[Any],
    languages: tuple[str, ...] = ("en", "ru"),
    max_turns: int | None = None,
    max_chars: int | None = None,
    max_rank: int | None = None,
    min_reviews: int = 1,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Convert in-memory rows. This function performs no I/O or downloads."""
    row_rejections: Counter[str] = Counter()
    parsed: list[Node] = []
    input_rows = 0
    for row in rows:
        input_rows += 1
        node, reason = parse_node(row)
        if reason:
            row_rejections[reason] += 1
        else:
            parsed.append(node)  # type: ignore[arg-type]

    occurrences: dict[str, list[Node]] = defaultdict(list)
    for node in parsed:
        occurrences[node.message_id].append(node)
    duplicate_ids = {message_id for message_id, values in occurrences.items() if len(values) > 1}
    global_nodes = {
        message_id: values[0]
        for message_id, values in occurrences.items()
        if message_id not in duplicate_ids
    }
    cross_tree_ids: set[str] = set()
    for node in parsed:
        if node.parent_id is None:
            continue
        parent = global_nodes.get(node.parent_id)
        if parent is not None and parent.tree_id != node.tree_id:
            cross_tree_ids.update((node.tree_id, parent.tree_id))
    trees: dict[str, list[Node]] = defaultdict(list)
    for node in parsed:
        trees[node.tree_id].append(node)

    tree_rejections: Counter[str] = Counter()
    candidate_rejections: Counter[str] = Counter()
    candidates_seen = 0
    valid_candidates = 0
    selected: list[Candidate] = []
    allowed_languages = set(languages)

    for tree_id in sorted(trees):
        nodes = sorted(trees[tree_id], key=lambda node: node.message_id)
        root, children, reason = validate_tree(nodes, global_nodes, duplicate_ids, cross_tree_ids)
        if reason:
            tree_rejections[reason] += 1
            continue

        eligible: list[Candidate] = []
        for path in leaf_paths(root, children):  # type: ignore[arg-type]
            candidates_seen += 1
            reason = candidate_rejection(
                path,
                allowed_languages,
                max_turns,
                max_chars,
                max_rank,
                min_reviews,
            )
            if reason:
                candidate_rejections[reason] += 1
                continue
            ranks = tuple(node.rank for node in path if node.role == "assistant")
            language = path[0].lang.strip().lower()
            eligible.append(Candidate(tree_id, path[-1].message_id, language, path, ranks))
            valid_candidates += 1
        if eligible:
            selected.append(min(eligible, key=selection_key))
        else:
            tree_rejections["no_valid_candidate"] += 1

    counts = {
        "input_rows": input_rows,
        "parsed_rows": len(parsed),
        "trees_seen": len(trees),
        "candidate_paths": candidates_seen,
        "valid_candidate_paths": valid_candidates,
        "output_records": len(selected),
    }
    rejections = {
        "rows": dict(sorted(row_rejections.items())),
        "trees": dict(sorted(tree_rejections.items())),
        "candidates": dict(sorted(candidate_rejections.items())),
    }
    return selected, {"counts": counts, "rejections": rejections}


def render_outputs(
    selected: list[Candidate], revision: str, split: str, config: str | None
) -> tuple[bytes, bytes]:
    output_lines: list[bytes] = []
    metadata_lines: list[bytes] = []
    source = {
        "dataset": SOURCE_DATASET,
        "revision": revision,
        "config": config,
        "split": split,
    }
    ordered = sorted(selected, key=lambda candidate: (candidate.tree_id, candidate.leaf_id))
    for line_number, candidate in enumerate(ordered, 1):
        record = {
            "messages": [
                {"role": "user" if node.role == "prompter" else "assistant", "content": node.text}
                for node in candidate.nodes
            ]
        }
        record_bytes = canonical_json(record)
        identity = canonical_json(
            {"source": source, "tree_id": candidate.tree_id, "leaf_id": candidate.leaf_id}
        )
        metadata = {
            "id": f"oasst1-{sha256(identity)[:24]}",
            "line": line_number,
            "message_tree_id": candidate.tree_id,
            "leaf_message_id": candidate.leaf_id,
            "message_ids": [node.message_id for node in candidate.nodes],
            "language": candidate.language,
            "content_sha256": sha256(record_bytes),
            "provenance": source,
        }
        output_lines.append(record_bytes)
        metadata_lines.append(canonical_json(metadata))
    output = b"".join(line + b"\n" for line in output_lines)
    metadata = b"".join(line + b"\n" for line in metadata_lines)
    return output, metadata


def nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Immutable Hugging Face dataset revision",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--languages", type=parse_languages, default=parse_languages("ru,en"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--max-turns", type=positive, help="Maximum messages in a path")
    parser.add_argument("--max-chars", type=positive, help="Maximum total characters in a path")
    parser.add_argument(
        "--min-reviews",
        type=nonnegative,
        default=1,
        help="Minimum review count for every assistant message",
    )
    parser.add_argument(
        "--max-rank",
        type=nonnegative,
        help="Inclusive maximum assistant rank (rank 0 is best)",
    )
    args = parser.parse_args()

    paths = [Path(args.output), Path(args.metadata_output), Path(args.manifest_output)]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("output, metadata, and manifest paths must differ")

    from datasets import load_dataset

    dataset = load_dataset(SOURCE_DATASET, split=args.split, revision=args.revision)
    config = getattr(getattr(dataset, "info", None), "config_name", None)
    selected, stats = convert_rows(
        dataset,
        languages=args.languages,
        max_turns=args.max_turns,
        max_chars=args.max_chars,
        max_rank=args.max_rank,
        min_reviews=args.min_reviews,
    )
    output, metadata = render_outputs(selected, args.revision, args.split, config)
    manifest = {
        "converter_version": CONVERTER_VERSION,
        "source": {
            "dataset": SOURCE_DATASET,
            "revision": args.revision,
            "config": config,
            "split": args.split,
        },
        "configuration": {
            "languages": list(args.languages),
            "max_turns": args.max_turns,
            "max_chars": args.max_chars,
            "max_rank": args.max_rank,
            "min_reviews": args.min_reviews,
            "rank_policy": "complete_then_mean_then_depth_then_leaf_id",
        },
        **stats,
        "files": {
            "output_sha256": sha256(output),
            "metadata_sha256": sha256(metadata),
        },
    }
    manifest_bytes = canonical_json(manifest) + b"\n"

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    paths[0].write_bytes(output)
    paths[1].write_bytes(metadata)
    paths[2].write_bytes(manifest_bytes)
    print(f"Wrote {len(selected)} conversations to {paths[0]}")


if __name__ == "__main__":
    main()
