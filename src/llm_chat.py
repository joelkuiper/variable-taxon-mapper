"""Prompt building and llama.cpp HTTP client helpers."""

from __future__ import annotations

import asyncio
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import aiohttp

from .utils import clean_text


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


class LlamaHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


async def llama_completion_async(
    prompt: Union[str, Iterable[Union[str, int]]],
    endpoint: str,
    *,
    timeout: float = 120.0,
    session: Optional[aiohttp.ClientSession] = None,
    **kwargs: Any,
) -> str:
    payload: Dict[str, Any] = {"prompt": prompt}
    payload.update(kwargs)

    close_session = False
    if session is None:
        timeout_cfg = aiohttp.ClientTimeout(
            total=None, sock_connect=10, sock_read=timeout
        )
        session = aiohttp.ClientSession(timeout=timeout_cfg)
        close_session = True
    try:
        request_timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=10, sock_read=timeout
        )
        async with session.post(
            endpoint,
            json=payload,
            timeout=request_timeout,
        ) as resp:
            if resp.status >= 400:
                raise LlamaHTTPError(resp.status, await resp.text())
            data = await resp.json(content_type=None)
            return data.get("content", "")
    finally:
        if close_session and session is not None:
            await session.close()


async def llama_completion_many(
    requests: Sequence[Tuple[Union[str, Iterable[Union[str, int]]], Dict[str, Any]]],
    endpoint: str,
    *,
    timeout: float = 120.0,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[str]:
    """Resolve multiple prompts concurrently while sharing an HTTP session."""

    if not requests:
        return []

    close_session = False
    if session is None:
        timeout_cfg = aiohttp.ClientTimeout(
            total=None, sock_connect=10, sock_read=timeout
        )
        session = aiohttp.ClientSession(timeout=timeout_cfg)
        close_session = True

    async def _run_single(
        prompt: Union[str, Iterable[Union[str, int]]], kwargs: Dict[str, Any]
    ) -> str:
        payload = dict(kwargs)
        payload.setdefault("session", session)
        payload.setdefault("timeout", timeout)
        return await llama_completion_async(prompt, endpoint, **payload)

    try:
        tasks = [
            asyncio.create_task(_run_single(prompt, kwargs))
            for prompt, kwargs in requests
        ]
        results = await asyncio.gather(*tasks)
    finally:
        if close_session and session is not None:
            await session.close()

    return list(results)


def llama_completion(
    prompt: Union[str, Iterable[Union[str, int]]],
    endpoint: str,
    *,
    timeout: float = 120.0,
    session: Optional[aiohttp.ClientSession] = None,
    **kwargs: Any,
) -> str:
    async def _runner() -> str:
        return await llama_completion_async(
            prompt,
            endpoint,
            timeout=timeout,
            session=session,
            **kwargs,
        )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_runner())
    else:
        if loop.is_running():
            raise RuntimeError(
                "llama_completion cannot be called while an event loop is running; "
                "use llama_completion_async instead."
            )
        return loop.run_until_complete(_runner())


def make_tree_match_prompt(
    tree_markdown_labels_only: str,
    item: Dict[str, Optional[str]],
    *,
    role_prefix: str = "<|im_start|>",
    role_suffix: str = "",
    eot: str = "<|im_end|>",
) -> str:
    tree_md = (tree_markdown_labels_only or "").strip()
    lab = clean_text(item.get("label"))
    nam = clean_text(item.get("name"))
    dat = clean_text(item.get("dataset"))
    desc = clean_text(item.get("description"))

    template = """\
        {role_prefix}system{role_suffix}
        # TASK
        • From the TREE (or SUGGESTIONS), choose **exactly one** concept that best matches the ITEM.
        • The TREE is a nested (indented) Markdown list where each bullet is: `- <concept label> [<optional short description>]`.
        • Concepts may include a short description in square brackets.
          Those descriptions are guidance only; **output must be the exact concept label (no description)**.
        • SUGGESTIONS were preselected based on similarity to ITEM, they are not exhaustive.
        • Output a single-line JSON, for example `{{"concept_label":"..."}}`.{eot}
        {role_prefix}user{role_suffix}
        {tree}

        # ITEM:
        **{item_label}** ({item_name})
        dataset: {item_dataset}
        description: {item_desc}{eot}
        {role_prefix}assistant{role_suffix}

    """
    return (
        dedent(template)
        .format(
            role_prefix=role_prefix,
            role_suffix=role_suffix,
            eot=eot,
            tree=tree_md,
            item_label=lab,
            item_dataset=dat,
            item_name=nam,
            item_desc=desc,
        )
        .strip()
    )
