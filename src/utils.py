from __future__ import annotations

import math
from typing import Any


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
