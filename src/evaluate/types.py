"""Type helpers used across the evaluation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence


ProgressHook = Callable[[int, int, Optional[int], float], None]


@dataclass
class PredictionJob:
    """Represents a unit of work sent through the prediction pipeline."""

    item: Dict[str, Optional[str]]
    slot_id: int
    metadata: Dict[str, Any]
    gold_labels: Optional[Sequence[str]] = None


