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
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional, Set

import pandas as pd
import requests
from tqdm.auto import tqdm

from src.taxonomy import build_name_maps_from_graph, build_taxonomy_graph
from src.utils import configure_logging


logger = logging.getLogger(__name__)


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
    temperature: float = 0.8,
    top_k: int = 20,
    top_p: float = 0.8,
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
Rewrite the provided definition as a SINGLE sentence fragment suitable for a description in a biomedical taxonomy.

Rules:
- Maximum {max_words} words.
- Expand biomedical acronyms when needed for clarity (e.g., spell out uncommon abbreviations).
""".strip()


@dataclass(frozen=True)
class SummaryContext:
    definition: str
    label: str
    path: str


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


def make_prompt(context: SummaryContext, max_words: int) -> str:
    sysmsg = SYSTEM_INSTRUCTIONS_TEMPLATE.format(max_words=max_words)

    user_lines = []
    if context.label:
        user_lines.append(f"Label: {context.label}")
    if context.path:
        user_lines.append(f"Full path to root: {context.path}")
    user_lines.append(f"Definition: {context.definition}")
    user = "\n".join(user_lines)

    # Chat-style markers commonly supported by llama.cpp chat templates
    return (
        "<|im_start|>system\n" + sysmsg + "\n<|im_end|>\n"
        "<|im_start|>user\n" + user + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def summarize_once(context: SummaryContext, endpoint: str, max_words: int) -> str:
    if not context.definition:
        return ""
    prompt = make_prompt(context, max_words)
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
    configure_logging(level=os.getenv("LOG_LEVEL", logging.INFO))

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
        logger.error("Input not found: %s", in_path)
        sys.exit(2)

    df = pd.read_csv(in_path, low_memory=False)
    if "definition" not in df.columns:
        logger.error("Input CSV lacks 'definition' column.")
        sys.exit(2)

    logger.info("Loaded %d rows from %s", len(df), in_path)

    name_to_path: Dict[str, str] = {}
    name_to_label_path: Dict[str, str] = {}
    if {"name", "parent"}.issubset(df.columns):
        work_df = df.copy()
        if "order" not in work_df.columns:
            work_df = work_df.copy()
            work_df["order"] = pd.NA
        try:
            G = build_taxonomy_graph(
                work_df,
                name_col="name",
                parent_col="parent",
                order_col="order",
            )
            _, raw_name_to_path = build_name_maps_from_graph(G)

            if "label" in work_df.columns:
                name_to_label: Dict[str, str] = {}
                for _, row in (
                    work_df[["name", "label"]].dropna(subset=["name"]).iterrows()
                ):
                    key = str(row["name"])
                    value = (
                        normalize(row.get("label"))
                        if isinstance(row.get("label"), str)
                        else ""
                    )
                    if key not in name_to_label or value:
                        name_to_label[key] = value

                for node, path_str in raw_name_to_path.items():
                    path_text = str(path_str)
                    name_to_path[node] = path_text
                    parts = [p.strip() for p in path_text.split("/") if p.strip()]
                    normalized_parts = []
                    for part in parts:
                        label_part = name_to_label.get(part, "")
                        normalized_parts.append(label_part if label_part else part)
                    name_to_label_path[node] = " / ".join(normalized_parts)
            else:
                name_to_path = raw_name_to_path
        except Exception as exc:
            logger.warning(
                "Unable to build taxonomy paths; continuing without them: %s",
                exc,
            )

    def _row_to_context(row) -> SummaryContext:
        definition = normalize(row.get("definition"))
        label_raw = row.get("label") if "label" in row else None
        if not isinstance(label_raw, str) or not label_raw.strip():
            label_raw = row.get("name") if "name" in row else None
        label = normalize(label_raw)
        name_val = row.get("name") if "name" in row else None
        path_lookup = ""
        if isinstance(name_val, str) and name_val:
            path_lookup = name_to_label_path.get(name_val) or name_to_path.get(
                name_val, ""
            )
        elif name_val is not None and pd.notna(name_val):
            key = str(name_val)
            path_lookup = name_to_label_path.get(key) or name_to_path.get(key, "")
        path = normalize(path_lookup)
        return SummaryContext(definition=definition, label=label, path=path)

    summary_map: Dict[SummaryContext, str] = {}
    contexts_to_summarize: Set[SummaryContext] = set()
    row_contexts: list[SummaryContext] = []

    for _, row in df.iterrows():
        ctx = _row_to_context(row)
        row_contexts.append(ctx)
        if ctx.definition:
            contexts_to_summarize.add(ctx)
        else:
            summary_map[ctx] = ""

    unique_contexts = sorted(
        contexts_to_summarize,
        key=lambda c: (c.definition, c.label, c.path),
    )

    logger.info(
        "Summarizing %d unique contexts using %d workers (max %d words)",
        len(unique_contexts),
        max_workers,
        max_words,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(summarize_once, ctx, endpoint, max_words): ctx
            for ctx in unique_contexts
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Summarizing"):
            ctx = futures[fut]
            try:
                summary_map[ctx] = fut.result()
            except Exception as e:
                # Leave empty on failure
                summary_map[ctx] = ""
                logger.warning("Summarization failed: %s", e)

    df["definition_summary"] = [summary_map.get(ctx, "") for ctx in row_contexts]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    n_nonempty = int((df["definition_summary"].str.len() > 0).sum())
    logger.info("Wrote summarized CSV to: %s", out_path)
    logger.info(
        "Summaries created: %d / %d rows (≤ %d words)",
        n_nonempty,
        len(df),
        max_words,
    )


if __name__ == "__main__":
    main()
