"""Utilities for writing CLI outputs in a variety of formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, cast

import pandas as pd
from pandas.io.common import get_handle

from vtm.utils import apply_output_prefix

OutputFormat = Literal["csv", "parquet", "feather", "json"]

_FORMAT_SUFFIXES: dict[OutputFormat, str] = {
    "csv": ".csv",
    "parquet": ".parquet",
    "feather": ".feather",
    "json": ".json",
}

_COMPRESSION_SUFFIXES: dict[str, str] = {
    "gzip": ".gz",
    "gz": ".gz",
    "bz2": ".bz2",
    "bzip2": ".bz2",
    "xz": ".xz",
    "lzma": ".xz",
    "zip": ".zip",
    "zstd": ".zst",
}

_KNOWN_SUFFIXES = {
    *{suffix.lower() for suffix in _FORMAT_SUFFIXES.values()},
    *set(_COMPRESSION_SUFFIXES.values()),
}


def _strip_known_suffixes(path: Path) -> Path:
    candidate = path
    while True:
        suffixes = candidate.suffixes
        if not suffixes:
            break
        last = suffixes[-1].lower()
        if last not in _KNOWN_SUFFIXES:
            break
        stem = Path(candidate.name).stem
        if not stem:
            break
        candidate = candidate.with_name(stem)
    return candidate


def normalise_output_path(
    path: Path, output_format: OutputFormat, compression: str | None
) -> Path:
    """Return ``path`` with suffixes adjusted for ``output_format`` and compression."""

    fmt = cast(OutputFormat, output_format.lower())
    base = _strip_known_suffixes(path)
    suffix = _FORMAT_SUFFIXES[fmt]
    candidate = base.with_suffix(suffix)
    if compression:
        compression_suffix = _COMPRESSION_SUFFIXES.get(compression.lower())
        if compression_suffix and not candidate.name.lower().endswith(compression_suffix):
            candidate = candidate.with_name(candidate.name + compression_suffix)
    return candidate


def build_sidecar_path(output_path: Path, kind: str, extension: str = ".json") -> Path:
    """Return the path for a sidecar file associated with ``output_path``."""

    base = _strip_known_suffixes(output_path)
    return output_path.with_name(f"{base.name}_{kind}{extension}")


def write_output(
    df: pd.DataFrame,
    path: Path,
    output_format: OutputFormat,
    *,
    compression: str | None = None,
    json_orient: str = "records",
    json_lines: bool = True,
    json_indent: int | None = None,
    keyword_columns: Iterable[str] = (),
    output_column_prefix: str | None = None,
) -> None:
    """Write ``df`` to ``path`` according to ``output_format``."""

    fmt = cast(OutputFormat, output_format.lower())

    keyword_columns = tuple(keyword_columns)
    if fmt == "csv":
        def _encode_standard_cell(value: Any) -> str:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            text = str(value)
            if any(ch in text for ch in [",", "\"", "\n", "\r"]):
                escaped = text.replace('"', '""')
                return f"\"{escaped}\""
            return text

        def _encode_keyword_cell(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, (list, tuple, set)):
                items = [str(item).strip() for item in value if item is not None]
            else:
                items = [str(value).strip()]
            items = [item for item in items if item]
            if not items:
                return ""
            joined = ",".join(items)
            if any("," in item for item in items):
                joined = joined.replace("\"", "\"\"")
                return f"\"\"\"{joined}\"\"\""
            return _encode_standard_cell(joined)

        header = [_encode_standard_cell(col) for col in df.columns]
        keyword_set = {
            apply_output_prefix(output_column_prefix or "", column)
            for column in keyword_columns
        }

        if compression:
            handle = get_handle(
                path, "w", compression=compression, encoding="utf-8", newline=""
            )
            fh = handle.handle
        else:
            handle = None
            fh = path.open("w", encoding="utf-8", newline="")
        try:
            fh.write(",".join(header) + "\n")
            for row in df.itertuples(index=False, name=None):
                encoded: list[str] = []
                for column, value in zip(df.columns, row):
                    if column in keyword_set:
                        encoded.append(_encode_keyword_cell(value))
                    else:
                        encoded.append(_encode_standard_cell(value))
                fh.write(",".join(encoded) + "\n")
        finally:
            if handle is not None:
                handle.close()
            else:
                fh.close()
        return

    if fmt == "parquet":
        kwargs = {"index": False}
        if compression:
            kwargs["compression"] = compression
        df.to_parquet(path, **kwargs)
        return

    if fmt == "feather":
        kwargs = {"index": False}
        if compression:
            kwargs["compression"] = compression
        df.to_feather(path, **kwargs)
        return

    if fmt == "json":
        kwargs = {
            "orient": json_orient,
            "lines": json_lines,
        }
        if json_indent is not None:
            kwargs["indent"] = json_indent
        if compression:
            kwargs["compression"] = compression
        df.to_json(path, **kwargs)
        return

    raise ValueError(f"Unsupported output format: {output_format}")


__all__ = [
    "OutputFormat",
    "build_sidecar_path",
    "normalise_output_path",
    "write_output",
]
