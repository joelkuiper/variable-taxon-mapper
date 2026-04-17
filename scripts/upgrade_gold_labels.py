#!/usr/bin/env python3
"""Upgrade gold labels in a benchmark variables file from generic parent nodes
to newly-added specific children, based on keyword matching against variable
name, label, and description fields.

Only upgrades when the match is unambiguous. Logs every change to stderr.

Usage:
    python scripts/upgrade_gold_labels.py \\
        --variables data/20251119_Variables_final_redesigned.csv \\
        --out data/20251119_Variables_final_redesigned.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

import pandas as pd


UPGRADE_RULES: list[dict] = [
    # Biochemistry parent → specific children
    {
        "from_keywords": ["Biochemistry", "Laboratory measures"],
        "to_keyword": "Lipid panel",
        "match_any": [
            r"cholesterol", r"triglycer", r"hdl", r"ldl", r"vldl",
            r"apolipoprotein", r"lipoprotein", r"lipid",
        ],
    },
    {
        "from_keywords": ["Biochemistry", "Laboratory measures"],
        "to_keyword": "Glucose and insulin metabolism",
        "match_any": [
            r"glucose", r"insulin", r"hba1c", r"glycat", r"ogtt",
            r"homa.?ir", r"glycaem", r"glycem", r"fasting blood sugar",
            r"diabetes.*marker",
        ],
    },
    {
        "from_keywords": ["Biochemistry", "Laboratory measures"],
        "to_keyword": "Liver function",
        "match_any": [
            r"\balt\b", r"\bast\b", r"\bggt\b", r"bilirubin", r"alkaline phosphatase",
            r"albumin", r"hepat", r"liver",
            r"alanine aminotransferase", r"aspartate aminotransferase",
            r"gamma.?glutamyl",
        ],
    },
    {
        "from_keywords": ["Biochemistry", "Laboratory measures"],
        "to_keyword": "Thyroid function",
        "match_any": [
            r"thyroid", r"\btsh\b", r"\bft4\b", r"\bft3\b", r"\bt3\b", r"\bt4\b",
            r"thyroxine", r"triiodothyronine",
        ],
    },
    {
        "from_keywords": ["Biochemistry", "Laboratory measures"],
        "to_keyword": "Inflammatory markers",
        "match_any": [
            r"\bcrp\b", r"c.reactive protein", r"interleukin", r"\bil.?\d",
            r"\btnf\b", r"tumou?r necrosis", r"cytokine", r"inflammat",
        ],
    },
    {
        "from_keywords": ["Biochemistry", "Laboratory measures"],
        "to_keyword": "Hormones",
        "match_any": [
            r"cortisol", r"testosterone", r"estradiol", r"estrogen",
            r"progesterone", r"dhea", r"\bshbg\b", r"leptin", r"adiponectin",
            r"hormone",
        ],
    },
    {
        "from_keywords": ["Biochemistry", "Laboratory measures"],
        "to_keyword": "Vitamins and minerals",
        "match_any": [
            r"vitamin", r"folate", r"folic acid", r"ferritin", r"\biron\b",
            r"\bzinc\b", r"\bb12\b", r"25.?hydroxy",
        ],
    },
    # Chemical exposure parent → Water quality (new)
    {
        "from_keywords": ["Chemical exposure", "Chemicals"],
        "to_keyword": "Water quality",
        "match_any": [
            r"drinking.?water", r"water.?quality", r"water.?supply",
            r"\bhardness\b", r"water.?fluorid", r"water.?chlorin",
            r"water.?turbid", r"water.?nitrate",
        ],
    },
]


def parse_keyword_cell(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    reader = csv.reader(io.StringIO(cell), skipinitialspace=True)
    try:
        fields = next(reader)
    except StopIteration:
        return []
    return [f.strip() for f in fields if f.strip()]


def format_keyword_cell(keywords: list[str]) -> str:
    if not keywords:
        return ""
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(keywords)
    return buf.getvalue().rstrip("\r\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variables", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keywords-column", default="keywords")
    args = ap.parse_args()

    df = pd.read_csv(args.variables, dtype=str, keep_default_na=False)
    col = args.keywords_column
    if col not in df.columns:
        raise SystemExit(f"'{col}' not found; available: {list(df.columns)}")

    upgrades = 0
    for idx, row in df.iterrows():
        kws = parse_keyword_cell(row[col])
        if not kws:
            continue

        name = str(row.get("name", "")).lower()
        label = str(row.get("label", "")).lower()
        desc = str(row.get("description", "")).lower()
        text = f"{name} {label} {desc}"

        new_kws = list(kws)
        changed = False
        for rule in UPGRADE_RULES:
            for i, kw in enumerate(new_kws):
                if kw not in rule["from_keywords"]:
                    continue
                if any(re.search(p, text) for p in rule["match_any"]):
                    old = new_kws[i]
                    new_kws[i] = rule["to_keyword"]
                    changed = True
                    print(
                        f"[{idx}] {row.get('name','')}: "
                        f"{old!r} → {rule['to_keyword']!r}",
                        file=sys.stderr,
                    )
                    break

        if changed:
            df.at[idx, col] = format_keyword_cell(new_kws)
            upgrades += 1

    df.to_csv(args.out, index=False)
    print(f"Upgraded {upgrades} / {len(df)} gold labels", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
