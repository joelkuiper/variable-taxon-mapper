"""Evaluation entry points and type aliases."""

from __future__ import annotations

from .benchmark import run_label_benchmark
from .collector import async_collect_predictions, collect_predictions
from .metrics import determine_match_type, is_correct_prediction
from .types import ProgressHook

__all__ = [
    "ProgressHook",
    "determine_match_type",
    "is_correct_prediction",
    "collect_predictions",
    "async_collect_predictions",
    "run_label_benchmark",
]
