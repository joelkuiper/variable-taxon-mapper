#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize the 'definition' column of data/Keywords.csv to <= 25 words
using a local llama.cpp HTTP endpoint, and write data/Keywords_summarized.csv.

Usage:
    python summarize_definitions.py \
        --in data/Keywords.csv \
        --out data/Keywords_summarized.csv \
        --endpoint http://127.0.0.1:8080/completions \
        --max-workers 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm


# -------------------------------
# HTTP client to llama.cpp
# -------------------------------


class LlamaHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def _make_adapter(pool_maxsize: int) -> HTTPAdapter:
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    return HTTPAdapter(
        pool_connections=pool_maxsize, pool_maxsize=pool_maxsize, max_retries=retry
    )


_tls = threading.local()


def _get_session(pool_maxsize: int = 64) -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        adapter = _make_adapter(pool_maxsize)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _tls.session = s
    return s


def llama_completion(
    endpoint: str,
    prompt: str,
    *,
    grammar: Optional[str] = None,
    temperature: float = 0.2,
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.0,
    n_predict: int = 96,
    timeout: float = 120.0,
    pool_maxsize: int = 64,
) -> str:
    """POST /completions to llama.cpp; returns 'content'."""
    s = _get_session(pool_maxsize)
    payload = {
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "n_predict": n_predict,
    }
    if grammar:
        payload["grammar"] = grammar
    resp = s.post(endpoint, json=payload, timeout=(10, timeout))
    if not resp.ok:
        raise LlamaHTTPError(resp.status_code, resp.text)
    return resp.json().get("content", "")


# -------------------------------
# Summarization logic
# -------------------------------


SYSTEM_INSTRUCTIONS = """\
You write concise biomedical definitions.
Rules:
- Return ONLY a single sentence fragment in plain text (no quotes, no JSON).
- Maximum 25 words. Prefer simple vocabulary. Remove noise and extraneous details.
- Keep essential meaning; avoid examples, citations, or abbreviations unless crucial.
""".strip()


def make_prompt(def_text: str) -> str:
    def_clean = normalize(def_text)
    user = f"Definition (messy): {def_clean}\n\nSummarize to <=25 words:"
    # Using chat-like markers (works fine with most llama.cpp chat templates)
    return (
        "<|im_start|>system\n" + SYSTEM_INSTRUCTIONS + "<|im_end|>\n"
        "<|im_start|>user\n" + user + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def normalize(s: Optional[str]) -> str:
    if s is None:
        return ""
    # Collapse whitespace and strip
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


def postprocess_25_words(s: str) -> str:
    """
    Safety trim to <= 25 words; also collapses spaces and strips surrounding quotes/backticks.
    """
    s = normalize(s)
    # Strip common wrappers accidentally emitted
    s = s.strip(" \"'`")
    words = s.split(" ")
    if len(words) > 25:
        s = " ".join(words[:25])
    return s


def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def summarize_once(def_text: str, endpoint: str, grammar_25w: str) -> str:
    if not def_text:
        return ""
    prompt = make_prompt(def_text)
    try:
        out = llama_completion(
            endpoint=endpoint,
            prompt=prompt,
            grammar=grammar_25w,
            temperature=0.1,  # stable
            top_k=0,  # greedy-ish with grammar
            top_p=1.0,
            min_p=0.0,
            n_predict=128,
        )
    except Exception as e:
        # Fallback: try once more without grammar; then trim
        try:
            out = llama_completion(
                endpoint=endpoint,
                prompt=prompt,
                grammar=None,
                temperature=0.2,
                top_k=40,
                top_p=0.95,
                n_predict=96,
            )
        except Exception:
            raise e
    return postprocess_25_words(out)


# -------------------------------
# Main: read → summarize → write
# -------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_csv", default="data/Keywords.csv", help="Input CSV path"
    )
    ap.add_argument(
        "--out",
        dest="out_csv",
        default="data/Keywords_summarized.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8080/completions",
        help="llama.cpp /completions URL",
    )
    ap.add_argument("--max-workers", type=int, default=8, help="Thread pool size")
    ap.add_argument(
        "--pool-maxsize", type=int, default=64, help="HTTP adapter pool size"
    )
    args = ap.parse_args()

    in_path = args.in_csv
    out_path = args.out_csv
    endpoint = args.endpoint
    max_workers = max(1, args.max_workers)

    if not os.path.exists(in_path):
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    # Read CSV
    df = pd.read_csv(in_path, low_memory=False)
    if "definition" not in df.columns:
        print("ERROR: Input CSV lacks 'definition' column.", file=sys.stderr)
        sys.exit(2)

    # Build cache for identical definitions (saves LLM calls)
    defs = df["definition"].fillna("").astype(str).map(normalize)
    uniq_map: Dict[str, str] = {}

    # Prepare work items
    unique_defs = sorted(set(defs.tolist()))
    # Skip blanks
    unique_defs = [d for d in unique_defs if d]

    # Summarize in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2txt = {
            ex.submit(summarize_once, d, endpoint, grammar_25w): d for d in unique_defs
        }
        for fut in tqdm(as_completed(fut2txt), total=len(fut2txt), desc="Summarizing"):
            d = fut2txt[fut]
            try:
                summary = fut.result()
            except Exception as e:
                # Leave empty on hard failure
                summary = ""
                print(
                    f"[WARN] Summarization failed for hash={hash_text(d)}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
            uniq_map[d] = summary

    # Map back to rows; keep empty for missing/failed
    df["definition_summary"] = defs.map(lambda s: uniq_map.get(s, ""))

    # Write output CSV (preserve all original columns + new summary)
    # Use utf-8, keep quoting minimal
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote summarized CSV to: {out_path}")
    # Small stats
    n_nonempty = int((df["definition_summary"].str.len() > 0).sum())
    print(f"Summaries created: {n_nonempty} / {len(df)} rows")


if __name__ == "__main__":
    main()
