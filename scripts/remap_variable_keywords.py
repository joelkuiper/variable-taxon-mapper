#!/usr/bin/env python3
"""Remap a variables CSV's keyword column from source-taxonomy names to
the cleaned-taxonomy names, using the provenance emitted by
transform_taxonomy.py.

Each row's `keywords` cell is a comma-separated list (with standard CSV
quoting for entries that contain commas, e.g.
``"Endocrine, nutritional and metabolic diseases (E00-E90)", Cats``).

Unmapped keywords are kept as-is and reported to stderr so a human can
triage them. Duplicates after mapping are deduped while preserving first
occurrence order.

Usage:
    python scripts/remap_variable_keywords.py \\
        --variables data/20251119_Variables_final.csv \\
        --provenance data/Keywords_redesigned_provenance.csv \\
        --out data/20251119_Variables_final_redesigned.csv \\
        [--keywords-column keywords]
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import pandas as pd


def parse_keyword_cell(cell: str) -> list[str]:
    """Parse a keywords cell as a mini-CSV row so quoted commas survive."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    reader = csv.reader(io.StringIO(cell), skipinitialspace=True)
    try:
        fields = next(reader)
    except StopIteration:
        return []
    return [f.strip() for f in fields if f.strip()]


def format_keyword_cell(keywords: list[str]) -> str:
    """Inverse: CSV-quote entries containing commas, join with ', '."""
    if not keywords:
        return ""
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(keywords)
    return buf.getvalue().rstrip("\r\n")


def build_name_map(prov_path: Path) -> dict[str, str]:
    prov = pd.read_csv(prov_path, dtype=str, keep_default_na=False).fillna("")
    missing = {"source_name", "current_name"} - set(prov.columns)
    if missing:
        raise SystemExit(f"provenance missing columns: {sorted(missing)}")
    return dict(zip(prov["source_name"], prov["current_name"]))


def remap_cell(cell: str, name_map: dict[str, str], unmapped: set[str]) -> str:
    original = parse_keyword_cell(cell)
    if not original:
        return ""
    seen: set[str] = set()
    new: list[str] = []
    for kw in original:
        mapped = name_map.get(kw)
        if mapped is None:
            unmapped.add(kw)
            mapped = kw
        if mapped not in seen:
            seen.add(mapped)
            new.append(mapped)
    return format_keyword_cell(new)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variables", type=Path, required=True)
    ap.add_argument("--provenance", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keywords-column", default="keywords")
    args = ap.parse_args()

    name_map = build_name_map(args.provenance)
    print(f"Loaded provenance with {len(name_map)} source → current mappings", file=sys.stderr)

    df = pd.read_csv(args.variables, dtype=str, keep_default_na=False)
    if args.keywords_column not in df.columns:
        raise SystemExit(
            f"'{args.keywords_column}' column not found; available: {list(df.columns)}"
        )
    print(f"Loaded {len(df)} variables from {args.variables}", file=sys.stderr)

    unmapped: set[str] = set()
    df[args.keywords_column] = df[args.keywords_column].apply(
        lambda cell: remap_cell(cell, name_map, unmapped)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}", file=sys.stderr)

    if unmapped:
        print(f"\n{len(unmapped)} keywords had no provenance entry (kept as-is):", file=sys.stderr)
        for kw in sorted(unmapped):
            print(f"  {kw!r}", file=sys.stderr)
    else:
        print("All keywords mapped cleanly.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
