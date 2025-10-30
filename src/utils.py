from __future__ import annotations

import math
import os
import random
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch


def clean_text(value: Any, empty="(empty)") -> str:
    """Normalize arbitrary values to a trimmed string or `empty`."""

    if value is None:
        return empty
    if isinstance(value, str):
        text = value.strip()
        return text if text else empty
    try:
        if isinstance(value, float) and math.isnan(value):
            return empty
    except Exception:
        # ``math.isnan`` may raise on non-numeric types; ignore and fall back.
        pass
    text = str(value).strip()
    return text if text else empty


def clean_str_or_none(v) -> Optional[str]:
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return s if s else None


def split_keywords_comma(s: Optional[str]) -> List[str]:
    if not isinstance(s, str):
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def set_global_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""

    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except (RuntimeError, ValueError):
            # Fallback when deterministic algorithms are unsupported for the current
            # device/kernel combination.
            pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Initialize application logging if it has not already been configured."""

    if isinstance(level, str):
        resolved_level = logging.getLevelName(level.upper())
        if isinstance(resolved_level, int):
            level = resolved_level
        else:
            raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def ensure_file_exists(path: Path, description: Optional[str] = None) -> Path:
    """Ensure ``path`` exists, raising a helpful :class:`FileNotFoundError`."""

    if not path.exists():
        desc = description or "file"
        raise FileNotFoundError(f"Required {desc} not found at {path}")
    return path
