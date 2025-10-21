"""Evaluation entry points and type aliases."""
from __future__ import annotations

from .benchmark import run_label_benchmark
from .metrics import determine_match_type, is_correct_prediction
from .types import ProgressHook

__all__ = [
    "ProgressHook",
    "determine_match_type",
    "is_correct_prediction",
    "run_label_benchmark",
]
