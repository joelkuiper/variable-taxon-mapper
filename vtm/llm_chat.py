"""Prompt building and OpenAI-compatible chat client helpers."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openai import AsyncOpenAI, OpenAI


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


_SYNC_CLIENTS: Dict[Tuple[str, str], OpenAI] = {}
_ASYNC_CLIENTS: Dict[Tuple[str, str], AsyncOpenAI] = {}
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


def _resolve_api_key(explicit: Optional[str]) -> str:
    if explicit is not None:
        candidate = explicit.strip()
        if candidate:
            return candidate
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    return "sk-no-key-required"


def _get_sync_client(endpoint: str, *, api_key: Optional[str] = None) -> OpenAI:
    base = _normalize_api_base(endpoint)
    resolved_key = _resolve_api_key(api_key)
    cache_key = (base, resolved_key)
    client = _SYNC_CLIENTS.get(cache_key)
    if client is None:
        client = OpenAI(base_url=base, api_key=resolved_key)
        _SYNC_CLIENTS[cache_key] = client
    return client


def _get_async_client(endpoint: str, *, api_key: Optional[str] = None) -> AsyncOpenAI:
    base = _normalize_api_base(endpoint)
    resolved_key = _resolve_api_key(api_key)
    cache_key = (base, resolved_key)
    client = _ASYNC_CLIENTS.get(cache_key)
    if client is None:
        client = AsyncOpenAI(base_url=base, api_key=resolved_key)
        _ASYNC_CLIENTS[cache_key] = client
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
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> str:
    client = _get_async_client(endpoint, api_key=api_key)
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
    api_key: Optional[str] = None,
) -> List[str]:
    """Resolve multiple chat prompts concurrently."""

    if not requests:
        return []

    client = _get_async_client(endpoint, api_key=api_key)

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
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> str:
    client = _get_sync_client(endpoint, api_key=api_key)
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

