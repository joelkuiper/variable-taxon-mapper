"""Prompt building and OpenAI-compatible chat client helpers."""

from __future__ import annotations

import asyncio
import logging
import os
from textwrap import dedent
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openai import AsyncOpenAI, OpenAI

from .utils import clean_text


logger = logging.getLogger(__name__)


GRAMMAR_RESPONSE = r"""
root      ::= obj
quote     ::= "\""

string ::=
  quote (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* quote


obj ::= ("{" quote "concept_label" quote ": " string "}")
"""


_SYNC_CLIENTS: Dict[str, OpenAI] = {}
_ASYNC_CLIENTS: Dict[str, AsyncOpenAI] = {}
_PROMPT_DEBUG_SHOWN = False


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


def _api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "sk-no-key-required")


def _get_sync_client(endpoint: str) -> OpenAI:
    base = _normalize_api_base(endpoint)
    client = _SYNC_CLIENTS.get(base)
    if client is None:
        client = OpenAI(base_url=base, api_key=_api_key())
        _SYNC_CLIENTS[base] = client
    return client


def _get_async_client(endpoint: str) -> AsyncOpenAI:
    base = _normalize_api_base(endpoint)
    client = _ASYNC_CLIENTS.get(base)
    if client is None:
        client = AsyncOpenAI(base_url=base, api_key=_api_key())
        _ASYNC_CLIENTS[base] = client
    return client


def _split_request_kwargs(kwargs: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    recognized_keys = {
        "frequency_penalty",
        "max_tokens",
        "presence_penalty",
        "response_format",
        "stop",
        "temperature",
        "top_p",
    }
    standard: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in recognized_keys:
            standard[key] = value
        else:
            extra[key] = value
    return standard, extra


async def llama_completion_async(
    messages: Sequence[Dict[str, Any]],
    endpoint: str,
    *,
    model: str,
    timeout: float = 120.0,
    **kwargs: Any,
) -> str:
    client = _get_async_client(endpoint)
    standard_kwargs, extra_body = _split_request_kwargs(dict(kwargs))
    response = await client.chat.completions.create(
        model=model,
        messages=list(messages),
        timeout=timeout,
        extra_body=extra_body or None,
        **standard_kwargs,
    )
    if not response.choices:
        raise RuntimeError("Chat completion returned no choices")
    message = response.choices[0].message
    if message is None or message.content is None:
        raise RuntimeError("Chat completion response missing message content")
    return message.content


async def llama_completion_many(
    requests: Sequence[Tuple[Sequence[Dict[str, Any]], Dict[str, Any]]],
    endpoint: str,
    *,
    model: str,
    timeout: float = 120.0,
) -> List[str]:
    """Resolve multiple chat prompts concurrently."""

    if not requests:
        return []

    client = _get_async_client(endpoint)

    async def _run_single(
        messages: Sequence[Dict[str, Any]], kwargs: Dict[str, Any]
    ) -> str:
        standard_kwargs, extra_body = _split_request_kwargs(dict(kwargs))
        response = await client.chat.completions.create(
            model=model,
            messages=list(messages),
            timeout=timeout,
            extra_body=extra_body or None,
            **standard_kwargs,
        )
        if not response.choices:
            raise RuntimeError("Chat completion returned no choices")
        message = response.choices[0].message
        if message is None or message.content is None:
            raise RuntimeError("Chat completion response missing message content")
        return message.content

    tasks = [asyncio.create_task(_run_single(messages, kwargs)) for messages, kwargs in requests]
    pending: Set[asyncio.Task[str]] = set(tasks)

    try:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_EXCEPTION
            )
            first_exc: Optional[BaseException] = None
            for task in done:
                exc = task.exception()
                if exc is not None:
                    first_exc = exc
                    break
            if first_exc is not None:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                raise first_exc
        return [task.result() for task in tasks]
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def llama_completion(
    messages: Sequence[Dict[str, Any]],
    endpoint: str,
    *,
    model: str,
    timeout: float = 120.0,
    **kwargs: Any,
) -> str:
    client = _get_sync_client(endpoint)
    standard_kwargs, extra_body = _split_request_kwargs(dict(kwargs))
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        timeout=timeout,
        extra_body=extra_body or None,
        **standard_kwargs,
    )
    if not response.choices:
        raise RuntimeError("Chat completion returned no choices")
    message = response.choices[0].message
    if message is None or message.content is None:
        raise RuntimeError("Chat completion response missing message content")
    return message.content


def _format_chat_messages(messages: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "?")).upper()
        content = str(message.get("content", ""))
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _print_prompt_once(messages: Sequence[Dict[str, Any]]) -> None:
    """Print the first LLM prompt for debugging."""

    global _PROMPT_DEBUG_SHOWN
    if not _PROMPT_DEBUG_SHOWN:
        _PROMPT_DEBUG_SHOWN = True
        formatted = _format_chat_messages(messages)
        logger.debug(
            "\n====== LLM PROMPT (one-time) ======\n%s\n====== END PROMPT ======\n",
            formatted,
        )


def make_tree_match_messages(
    tree_markdown_labels_only: str,
    item: Dict[str, Optional[str]],
) -> List[Dict[str, str]]:
    tree_md = (tree_markdown_labels_only or "").strip()
    lab = clean_text(item.get("label"))
    nam = clean_text(item.get("name"))
    dat = clean_text(item.get("dataset"))
    desc = clean_text(item.get("description"))

    system_content = dedent(
        """\
        # TASK
        • From the TREE (or SUGGESTIONS), choose **exactly one** concept that best matches the ITEM.
        • Prefer the most specific matching child, if present; only choose the parent if no child fits.
        • The TREE is a nested (indented) Markdown list where each bullet is: `- <concept label> [<optional short description>]`.
        • Concepts may include a short description in square brackets.
          Those descriptions are guidance only; **output must be the exact concept label (no description)**.
        • SUGGESTIONS were preselected based on similarity to ITEM, they are not exhaustive.
        • Output a single-line JSON, for example `{"concept_label":"..."}`.
        """
    ).strip()

    user_content = dedent(
        """\
        {tree}

        # ITEM:
        **{item_label}** ({item_name})
        dataset: {item_dataset}
        description: {item_desc}
        """
    ).format(
        tree=tree_md,
        item_label=lab,
        item_name=nam,
        item_dataset=dat,
        item_desc=desc,
    ).strip()

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
