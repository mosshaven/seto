"""Data quality filtering pipeline for Seto pretraining."""

import hashlib
import re
from typing import List, Tuple, Optional


# Language detection via character ratios
LANG_PATTERNS = {
    "ru": re.compile(r"[а-яА-ЯёЁ]"),
    "en": re.compile(r"[a-zA-Z]"),
    "uk": re.compile(r"[а-яА-ЯёЁіІїЇґҐ]"),
    "de": re.compile(r"[äöüÄÖÜß]"),
    "fr": re.compile(r"[àâéèêëïîôùûüÿçœæ]"),
}

QUALITY_RULES = {
    "min_length": 100,
    "max_length": 1000000,
    "min_word_count": 20,
    "max_alpha_ratio": 0.95,
    "min_alpha_ratio": 0.30,
    "max_digit_ratio": 0.50,
    "max_special_ratio": 0.30,
    "max_line_repetition": 0.50,
    "min_avg_word_length": 2.0,
    "max_avg_word_length": 15.0,
}


def detect_language(text: str) -> str:
    counts = {}
    for lang, pattern in LANG_PATTERNS.items():
        counts[lang] = len(pattern.findall(text))

    if not counts:
        return "unknown"

    total = sum(counts.values())
    if total == 0:
        return "unknown"

    best = max(counts, key=counts.get)
    if counts[best] / total < 0.3:
        return "unknown"
    return best


def compute_quality_score(text: str) -> float:
    if not text or not text.strip():
        return 0.0

    score = 1.0
    words = text.split()
    n_chars = len(text)
    n_words = len(words)

    # Length checks
    if n_chars < QUALITY_RULES["min_length"]:
        return 0.0
    if n_chars > QUALITY_RULES["max_length"]:
        return 0.0
    if n_words < QUALITY_RULES["min_word_count"]:
        return 0.0

    # Alpha ratio
    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / max(1, n_chars)
    if alpha_ratio > QUALITY_RULES["max_alpha_ratio"]:
        score *= 0.5
    if alpha_ratio < QUALITY_RULES["min_alpha_ratio"]:
        return 0.0

    # Digit ratio
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / max(1, n_chars)
    if digit_ratio > QUALITY_RULES["max_digit_ratio"]:
        score *= 0.5

    # Special char ratio
    special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_ratio = special_count / max(1, n_chars)
    if special_ratio > QUALITY_RULES["max_special_ratio"]:
        score *= 0.5

    # Line repetition
    lines = text.split("\n")
    if lines:
        unique_lines = set(l.strip() for l in lines if l.strip())
        rep_ratio = 1.0 - len(unique_lines) / max(1, len(lines))
        if rep_ratio > QUALITY_RULES["max_line_repetition"]:
            score *= 0.3

    # Word length
    if words:
        avg_wl = sum(len(w) for w in words) / len(words)
        if avg_wl < QUALITY_RULES["min_avg_word_length"]:
            score *= 0.7
        if avg_wl > QUALITY_RULES["max_avg_word_length"]:
            score *= 0.7

    # Sentence structure bonus
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) > 3:
        score *= 1.1

    # Paragraph structure bonus
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        score *= 1.05

    return min(1.0, score)


def dedup_hash(texts: List[str], threshold: float = 0.9) -> List[Tuple[int, float]]:
    seen_hashes = {}
    results = []

    for i, text in enumerate(texts):
        # SimHash-like: split into shingles and hash
        shingles = set()
        words = text.split()
        for j in range(len(words) - 2):
            shingle = " ".join(words[j:j+3])
            shingles.add(hash(shingle))

        if not shingles:
            results.append((i, 1.0))
            continue

        text_hash = hash(frozenset(shingles))

        is_dup = False
        for seen_hash, seen_idx in seen_hashes.items():
            # Simple Jaccard estimate
            overlap = len(shingles & seen_hashes[seen_hash][1])
            total = len(shingles | seen_hashes[seen_hash][1])
            if total > 0 and overlap / total > threshold:
                is_dup = True
                break

        if not is_dup:
            seen_hashes[text_hash] = (i, shingles)
            results.append((i, 1.0))
        else:
            results.append((i, 0.0))

    return results


def filter_text(
    text: str,
    min_quality: float = 0.3,
    allowed_languages: Optional[List[str]] = None,
) -> Tuple[bool, float, str]:
    quality = compute_quality_score(text)
    if quality < min_quality:
        return False, quality, "low_quality"

    lang = detect_language(text)
    if allowed_languages and lang not in allowed_languages:
        return False, quality, f"wrong_language:{lang}"

    return True, quality, lang


def filter_dataset(
    texts: List[str],
    min_quality: float = 0.3,
    allowed_languages: Optional[List[str]] = None,
    deduplicate: bool = True,
) -> List[Tuple[int, str, float, str]]:
    results = []
    for i, text in enumerate(texts):
        passed, quality, reason = filter_text(text, min_quality, allowed_languages)
        if passed:
            results.append((i, text, quality, reason))

    if deduplicate:
        dedup_results = dedup_hash([r[1] for r in results])
        results = [
            (r[0], r[1], r[2], r[3])
            for r, (idx, score) in zip(results, dedup_results)
            if score > 0
        ]

    return results
