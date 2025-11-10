"""String similarity helpers backed by RapidFuzz."""

from __future__ import annotations

from rapidfuzz import fuzz


def normalize_similarity_text(value: str) -> str:
    """Normalize text for similarity comparisons."""

    if not value:
        return ""
    return value.strip().casefold()


def normalized_score(score: float) -> float:
    """Convert a RapidFuzz percentage score into a 0-1 float."""

    return float(score) / 100.0


def normalized_token_sort_ratio(a: str, b: str) -> float:
    """Return the normalized token sort ratio in the ``[0, 1]`` range."""

    norm_a = normalize_similarity_text(a)
    norm_b = normalize_similarity_text(b)
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0
    return normalized_score(fuzz.token_sort_ratio(norm_a, norm_b, score_cutoff=0.0))


def normalized_token_set_ratio(a: str, b: str) -> float:
    """Return the normalized token set ratio in the ``[0, 1]`` range."""

    norm_a = normalize_similarity_text(a)
    norm_b = normalize_similarity_text(b)
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0
    return normalized_score(fuzz.token_set_ratio(norm_a, norm_b, score_cutoff=0.0))
