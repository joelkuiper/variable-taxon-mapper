"""Prompt building and OpenAI-compatible chat client helpers."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam
import httpx

from vtm.config import HttpConfig, LLMConfig


logger = logging.getLogger(__name__)


MATCH_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "concept_labels": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 1,
        },
    },
    "required": ["concept_labels"],
    "additionalProperties": False,
}


def json_schema_response_format(
    name: str, schema: Mapping[str, Any]
) -> Dict[str, Any]:
    """Construct a ``response_format`` payload for JSON schema constrained output."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": copy.deepcopy(schema),
        },
    }


DEFAULT_MATCH_RESPONSE_FORMAT = json_schema_response_format(
    "match_response", MATCH_RESPONSE_SCHEMA
)


_SYNC_CLIENTS: Dict[Tuple[str, str], OpenAI] = {}
_ASYNC_CLIENTS: Dict[Tuple[str, str], AsyncOpenAI] = {}
_PROMPT_DEBUG_SHOWN = False


def _resolve_timeout_components(
    *,
    timeout: float | None,
    http_cfg: HttpConfig | None,
    llm_cfg: LLMConfig | None,
) -> tuple[float, httpx.Timeout | float]:
    from vtm.evaluate.parallelism import sock_read_timeout

    resolved_read: float
    if http_cfg is not None and llm_cfg is not None:
        resolved_read = float(sock_read_timeout(http_cfg, llm_cfg))
    elif timeout is not None:
        resolved_read = float(timeout)
    elif llm_cfg is not None:
        resolved_read = float(llm_cfg.n_predict)
    else:
        resolved_read = 120.0

    if http_cfg is None and llm_cfg is not None and timeout is None:
        resolved_read = max(resolved_read, 64.0)
    elif timeout is not None:
        resolved_read = max(resolved_read, 0.0)

    connect_timeout: float | None = None
    if http_cfg is not None:
        connect_timeout = float(http_cfg.sock_connect)
        resolved_read = max(resolved_read, float(http_cfg.sock_read_floor))

    if connect_timeout is not None:
        client_timeout: httpx.Timeout | float = httpx.Timeout(
            connect=connect_timeout,
            read=resolved_read,
            write=resolved_read,
            pool=connect_timeout,
        )
    else:
        client_timeout = resolved_read if timeout is not None else resolved_read

    return resolved_read, client_timeout


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
        "json_schema",
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
    messages: Sequence[ChatCompletionMessageParam],
    endpoint: str,
    *,
    model: str,
    timeout: float | None = None,
    api_key: Optional[str] = None,
    http_cfg: HttpConfig | None = None,
    llm_cfg: LLMConfig | None = None,
    **kwargs: Any,
) -> str:
    read_timeout, client_timeout = _resolve_timeout_components(
        timeout=timeout, http_cfg=http_cfg, llm_cfg=llm_cfg
    )
    client = _get_async_client(endpoint, api_key=api_key).with_options(
        timeout=client_timeout
    )
    standard_kwargs, extra_body = _split_request_kwargs(dict(kwargs))
    request = client.chat.completions.create(
        model=model,
        messages=list(messages),
        extra_body=extra_body or None,
        timeout=client_timeout,
        **standard_kwargs,
    )
    response = await asyncio.wait_for(request, read_timeout)
    if not response.choices:
        raise RuntimeError("Chat completion returned no choices")
    message = response.choices[0].message
    if message is None or message.content is None:
        raise RuntimeError("Chat completion response missing message content")
    return message.content


async def llama_completion_many(
    requests: Sequence[Tuple[Sequence[ChatCompletionMessageParam], Dict[str, Any]]],
    endpoint: str,
    *,
    model: str,
    timeout: float | None = None,
    api_key: Optional[str] = None,
    http_cfg: HttpConfig | None = None,
    llm_cfg: LLMConfig | None = None,
) -> List[str]:
    """Resolve multiple chat prompts concurrently."""

    if not requests:
        return []

    read_timeout, client_timeout = _resolve_timeout_components(
        timeout=timeout, http_cfg=http_cfg, llm_cfg=llm_cfg
    )

    client = _get_async_client(endpoint, api_key=api_key).with_options(
        timeout=client_timeout
    )

    async def _run_single(
        messages: Sequence[ChatCompletionMessageParam], kwargs: Dict[str, Any]
    ) -> str:
        standard_kwargs, extra_body = _split_request_kwargs(dict(kwargs))
        request = client.chat.completions.create(
            model=model,
            messages=list(messages),
            extra_body=extra_body or None,
            timeout=client_timeout,
            **standard_kwargs,
        )
        response = await asyncio.wait_for(request, read_timeout)
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
    messages: Sequence[ChatCompletionMessageParam],
    endpoint: str,
    *,
    model: str,
    timeout: float | None = None,
    api_key: Optional[str] = None,
    http_cfg: HttpConfig | None = None,
    llm_cfg: LLMConfig | None = None,
    **kwargs: Any,
) -> str:
    read_timeout, client_timeout = _resolve_timeout_components(
        timeout=timeout, http_cfg=http_cfg, llm_cfg=llm_cfg
    )
    client = _get_sync_client(endpoint, api_key=api_key).with_options(
        timeout=client_timeout
    )
    standard_kwargs, extra_body = _split_request_kwargs(dict(kwargs))
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        extra_body=extra_body or None,
        timeout=client_timeout,
        **standard_kwargs,
    )
    # ``httpx`` performs synchronous waits, but we defensively mirror the async
    # interface's minimum timeout expectations.
    if read_timeout <= 0:
        raise TimeoutError("Configured read timeout must be positive")
    if not response.choices:
        raise RuntimeError("Chat completion returned no choices")
    message = response.choices[0].message
    if message is None or message.content is None:
        raise RuntimeError("Chat completion response missing message content")
    return message.content
