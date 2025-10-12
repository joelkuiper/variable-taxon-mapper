from __future__ import annotations

import math
import numpy as np
from typing import Any, List, Optional


def clean_text(value: Any) -> str:
    """Normalize arbitrary values to a trimmed string or ``"(empty)"``."""

    if value is None:
        return "(empty)"
    if isinstance(value, str):
        text = value.strip()
        return text if text else "(empty)"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "(empty)"
    except Exception:
        # ``math.isnan`` may raise on non-numeric types; ignore and fall back.
        pass
    text = str(value).strip()
    return text if text else "(empty)"


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
