from __future__ import annotations

import math
import os
import random
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import torch
import pandas as pd


logger = logging.getLogger(__name__)


def apply_output_prefix(prefix: Optional[str], name: str) -> str:
    """Return ``name`` prefixed with ``prefix`` when provided."""

    return f"{prefix}{name}" if prefix else name


def resolve_prefixed_column(
    columns: Iterable[str], name: str, prefix: Optional[str] = None
) -> Optional[str]:
    """Resolve ``name`` against ``columns`` with an optional prefix fallback."""

    if prefix:
        candidate = f"{prefix}{name}"
        if candidate in columns:
            return candidate
    if name in columns:
        return name
    target = name.lower()
    for column in columns:
        if str(column).lower().endswith(target):
            return column
    return None

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


def resolve_path(
    base_path: Path | None,
    default: Path,
    override: Path | None,
) -> Path:
    """Resolve an override path against an optional ``base_path``.

    Parameters
    ----------
    base_path:
        The directory from which relative overrides should be resolved. When
        ``None`` relative overrides are resolved relative to the current working
        directory.
    default:
        The fallback path used when ``override`` is ``None``.
    override:
        An optional path provided by the caller.

    Returns
    -------
    Path
        The resolved absolute path chosen using the override (when provided)
        or the default.
    """

    if override is None:
        candidate = default
    elif override.is_absolute():
        candidate = override
    elif base_path is not None:
        candidate = base_path / override
    else:
        candidate = override
    return candidate.resolve()


def load_table(
    path: str | Path,
    *,
    csv_engine: str | None = "pyarrow",
    **read_kwargs: Any,
) -> pd.DataFrame:
    """Load a tabular dataset from ``path`` using an appropriate backend.

    Parameters
    ----------
    path:
        File path pointing to a CSV/Parquet/Feather dataset. Compressed CSV files
        (e.g., ``.csv.gz``) are automatically handled.
    csv_engine:
        Preferred pandas engine for CSV loading. Defaults to ``"pyarrow"`` when
        available, falling back to pandas' default if the engine is unavailable.
        Pass ``None`` to rely on pandas' default behaviour directly.
    **read_kwargs:
        Additional keyword arguments forwarded to the underlying pandas loader.

    Returns
    -------
    pandas.DataFrame
        The loaded table.
    """

    resolved_path = Path(path)
    suffixes = [s.lower() for s in resolved_path.suffixes]
    suffix = suffixes[-1] if suffixes else ""
    if suffix == ".gz" and len(suffixes) > 1:
        suffix = suffixes[-2]

    if suffix in {".parquet"}:
        kwargs = dict(read_kwargs)
        for key in ("low_memory", "dtype"):
            kwargs.pop(key, None)
        return pd.read_parquet(resolved_path, **kwargs)

    if suffix in {".feather", ".ft"}:
        kwargs = dict(read_kwargs)
        for key in ("low_memory", "dtype"):
            kwargs.pop(key, None)
        return pd.read_feather(resolved_path, **kwargs)

    kwargs = dict(read_kwargs)
    engine = csv_engine
    if engine is not None:
        try:
            return pd.read_csv(resolved_path, engine=engine, **kwargs)
        except (ImportError, ValueError) as exc:
            logger.debug(
                "Falling back to pandas default CSV engine for %s due to error: %s",
                resolved_path,
                exc,
            )
            engine = None

    if engine is None and "engine" in kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k != "engine"}
    return pd.read_csv(resolved_path, **kwargs)
