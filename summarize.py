#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize the 'definition' column of data/Keywords.csv to <= N words
using a local llama.cpp HTTP endpoint, and write data/Keywords_summarized.csv.

Usage:
    python summarize_definitions.py \
        --in data/Keywords.csv \
        --out data/Keywords_summarized.csv \
        --endpoint http://127.0.0.1:8080/completions \
        --max-words 25 \
        --max-workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import pandas as pd
import requests
from tqdm.auto import tqdm


# -------------------------------
# llama.cpp client (minimal)
# -------------------------------

_SESSION: Optional[requests.Session] = None


def get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def llama_completion(
    endpoint: str,
    prompt: str,
    *,
    temperature: float = 0.2,
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.0,
    n_predict: int = 96,
    timeout: float = 120.0,
) -> str:
    """POST /completions to llama.cpp; returns 'content' text."""
    s = get_session()
    payload = {
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "n_predict": n_predict,
    }
    r = s.post(endpoint, json=payload, timeout=(10, timeout))
    r.raise_for_status()
    return r.json().get("content", "")


# -------------------------------
# Prompting & helpers
# -------------------------------

SYSTEM_INSTRUCTIONS_TEMPLATE = """\
You are a biomedical lexicographer.
Rewrite the provided definition as a SINGLE sentence fragment suitable for a description in a taxonomy.

Rules:
- Maximum {max_words} words.
- Expand uncommon abbreviations only if needed for clarity.
""".strip()


def normalize(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def trim_to_words(s: str, max_words: int) -> str:
    s = normalize(s).strip(" \"'`")
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    return s


def make_prompt(def_text: str, max_words: int) -> str:
    sysmsg = SYSTEM_INSTRUCTIONS_TEMPLATE.format(max_words=max_words)
    user = f"Original definition: {normalize(def_text)}\n"
    # Chat-style markers commonly supported by llama.cpp chat templates
    return (
        "<|im_start|>system\n" + sysmsg + "\n<|im_end|>\n"
        "<|im_start|>user\n" + user + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def summarize_once(def_text: str, endpoint: str, max_words: int) -> str:
    if not def_text:
        return ""
    prompt = make_prompt(def_text, max_words)
    out = llama_completion(
        endpoint=endpoint,
        prompt=prompt,
        temperature=0.8,
        top_k=20,
        top_p=0.8,
        min_p=0.0,
        n_predict=128,
    )

    return trim_to_words(out, max_words)


# -------------------------------
# Main
# -------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_csv", default="data/Keywords.csv", help="Input CSV"
    )
    ap.add_argument(
        "--out",
        dest="out_csv",
        default="data/Keywords_summarized.csv",
        help="Output CSV",
    )
    ap.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8080/completions",
        help="llama.cpp /completions URL",
    )
    ap.add_argument("--max-words", type=int, default=25, help="Max words per summary")
    ap.add_argument("--max-workers", type=int, default=8, help="Thread pool size")
    args = ap.parse_args()

    in_path = args.in_csv
    out_path = args.out_csv
    endpoint = args.endpoint
    max_words = max(1, args.max_words)
    max_workers = max(1, args.max_workers)

    if not os.path.exists(in_path):
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(in_path, low_memory=False)
    if "definition" not in df.columns:
        print("ERROR: Input CSV lacks 'definition' column.", file=sys.stderr)
        sys.exit(2)

    # Deduplicate work by definition text
    defs = df["definition"].fillna("").astype(str).map(normalize)
    unique_defs = sorted({d for d in defs if d})
    summary_map: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(summarize_once, d, endpoint, max_words): d for d in unique_defs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Summarizing"):
            d = futures[fut]
            try:
                summary_map[d] = fut.result()
            except Exception as e:
                # Leave empty on failure
                summary_map[d] = ""
                print(f"[WARN] Summarization failed: {e}", file=sys.stderr)

    df["definition_summary"] = defs.map(lambda s: summary_map.get(s, ""))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    n_nonempty = int((df["definition_summary"].str.len() > 0).sum())
    print(f"Wrote summarized CSV to: {out_path}")
    print(f"Summaries created: {n_nonempty} / {len(df)} rows (≤ {max_words} words)")


if __name__ == "__main__":
    main()
