from __future__ import annotations

import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import typer
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from tqdm.auto import tqdm

from vtm.taxonomy import build_name_maps_from_graph, build_taxonomy_graph

from .app import app, logger


_CLIENTS: Dict[str, OpenAI] = {}


def _normalize_api_base(endpoint: str) -> str:
    base = (endpoint or "").strip()
    if not base:
        raise ValueError("Endpoint must be a non-empty string")
    base = base.rstrip("/")
    if base.endswith("/completions"):
        base = base[: -len("/completions")]
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _get_client(endpoint: str) -> OpenAI:
    base = _normalize_api_base(endpoint)
    client = _CLIENTS.get(base)
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY", "sk-no-key-required")
        client = OpenAI(base_url=base, api_key=api_key)
        _CLIENTS[base] = client
    return client


def _split_request_kwargs(kwargs: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    recognized = {}
    extra = {}
    recognized_keys = {
        "frequency_penalty",
        "max_tokens",
        "presence_penalty",
        "response_format",
        "stop",
        "temperature",
        "top_p",
    }
    for key, value in kwargs.items():
        if key in recognized_keys:
            if value is not None:
                recognized[key] = value
        elif value is not None:
            extra[key] = value
    return recognized, extra


def chat_completion(
    *,
    endpoint: str,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    timeout: float = 120.0,
    **kwargs: Any,
) -> str:
    client = _get_client(endpoint).with_options(timeout=timeout)
    standard_kwargs, extra_body = _split_request_kwargs(dict(kwargs))
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        extra_body=extra_body or None,
        **standard_kwargs,
    )
    if not response.choices:
        raise RuntimeError("LLM returned no choices for the chat completion request")
    message = response.choices[0].message
    if message is None or message.content is None:
        raise RuntimeError("Chat completion response did not include message content")
    return message.content


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


def normalize(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def trim_to_words(s: str, max_words: int) -> str:
    s = normalize(s).strip(" \"'`")
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    return s


def make_messages(
    context: SummaryContext, max_words: int
) -> list[ChatCompletionMessageParam]:
    sysmsg = SYSTEM_INSTRUCTIONS_TEMPLATE.format(max_words=max_words)

    user_lines: List[str] = []
    if context.label:
        user_lines.append(f"Label: {context.label}")
    if context.path:
        user_lines.append(f"Full path to root: {context.path}")
    user_lines.append(f"Definition: {context.definition}")
    user = "\n".join(user_lines)

    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": sysmsg,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": user,
    }
    return [system_message, user_message]


def summarize_once(
    context: SummaryContext,
    endpoint: str,
    model: str,
    max_words: int,
) -> str:
    if not context.definition:
        return ""
    messages = make_messages(context, max_words)
    out = chat_completion(
        endpoint=endpoint,
        model=model,
        messages=messages,
        temperature=0.8,
        top_k=20,
        top_p=0.8,
        min_p=0.0,
        n_predict=128,
    )

    return trim_to_words(out, max_words)


def _iter_row_contexts(df: pd.DataFrame) -> Iterable[Tuple[int, SummaryContext]]:
    name_to_path: Dict[str, str] = {}
    name_to_label_path: Dict[str, str] = {}
    if {"name", "parent"}.issubset(df.columns):
        work_df = df.copy()
        if "order" not in work_df.columns:
            work_df = work_df.copy()
            work_df["order"] = pd.NA
        try:
            graph = build_taxonomy_graph(
                work_df,
                name_col="name",
                parent_col="parent",
                order_col="order",
            )
            _, raw_name_to_path = build_name_maps_from_graph(graph)

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
                    normalized_parts: List[str] = []
                    for part in parts:
                        label_part = name_to_label.get(part, "")
                        normalized_parts.append(label_part if label_part else part)
                    name_to_label_path[node] = " / ".join(normalized_parts)
            else:
                name_to_path = raw_name_to_path
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Unable to build taxonomy paths; continuing without them: %s",
                exc,
            )

    for idx, row in df.iterrows():
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
        yield idx, SummaryContext(definition=definition, label=label, path=path)


@app.command("summarize")
def summarize_command(
    input_csv: Path = typer.Option(
        Path("data/Keywords.csv"),
        "--in",
        help="Input keywords CSV path.",
        path_type=Path,
    ),
    output_csv: Path = typer.Option(
        Path("data/Keywords_summarized.csv"),
        "--out",
        help="Output CSV path.",
        path_type=Path,
    ),
    endpoint: str = typer.Option(
        "http://127.0.0.1:8080/v1",
        "--endpoint",
        help="OpenAI-compatible API base URL (e.g. http://127.0.0.1:8080/v1)",
    ),
    model: str = typer.Option(
        "gpt-3.5-turbo",
        "--model",
        help="Chat completion model name.",
    ),
    max_words: int = typer.Option(25, "--max-words", help="Maximum words per summary."),
    max_workers: int = typer.Option(8, "--max-workers", help="Thread pool size."),
) -> None:
    """Summarize keyword definitions using an OpenAI-compatible endpoint."""

    if not input_csv.exists():
        logger.error("Input not found: %s", input_csv)
        raise typer.Exit(code=2)

    df = pd.read_csv(input_csv, low_memory=False)
    if "definition" not in df.columns:
        logger.error("Input CSV lacks 'definition' column.")
        raise typer.Exit(code=2)

    logger.info("Loaded %d rows from %s", len(df), input_csv)

    max_words = max(1, max_words)
    max_workers = max(1, max_workers)

    summary_map: Dict[SummaryContext, str] = {}
    contexts_to_summarize: Set[SummaryContext] = set()
    row_contexts: List[SummaryContext] = []

    for _, ctx in _iter_row_contexts(df):
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(summarize_once, ctx, endpoint, model, max_words): ctx
            for ctx in unique_contexts
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Summarizing"
        ):
            ctx = futures[fut]
            try:
                summary_map[ctx] = fut.result()
            except Exception as exc:  # pragma: no cover - LLM failure
                summary_map[ctx] = ""
                logger.warning("Summarization failed: %s", exc)

    df["definition_summary"] = [summary_map.get(ctx, "") for ctx in row_contexts]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    n_nonempty = int((df["definition_summary"].str.len() > 0).sum())
    logger.info("Wrote summarized CSV to: %s", output_csv)
    logger.info(
        "Summaries created: %d / %d rows (≤ %d words)",
        n_nonempty,
        len(df),
        max_words,
    )
