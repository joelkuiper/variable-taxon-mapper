#!/usr/bin/env python3
"""Copy `definition_summary` into `definition` for every row that has a
non-empty summary. Reads the summarized CSV, writes the promoted CSV.

Usage:
    python scripts/promote_summaries.py \\
        --in  data/Keywords_redesigned_summarized.csv \\
        --out data/Keywords_redesigned.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False).fillna("")
    if "definition_summary" not in df.columns:
        raise SystemExit(f"'definition_summary' column not found in {args.input}")
    mask = df["definition_summary"].str.strip() != ""
    df.loc[mask, "definition"] = df.loc[mask, "definition_summary"]
    df.to_csv(args.out, index=False)
    print(f"Promoted {int(mask.sum())} / {len(df)} summaries into definition column")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
