"""Prompt building and llama.cpp HTTP client helpers."""

from __future__ import annotations

import asyncio
import math
from textwrap import dedent
from typing import Any, Dict, Iterable, Optional, Union

import aiohttp


GRAMMAR_RESPONSE = r"""
root      ::= obj
quote     ::= "\""

string ::=
  quote (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )+ quote

obj ::= ("{" quote "node_label" quote ": " string ","
             quote "rationale" quote ": " string "}")
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
    def _clean_text(v) -> str:
        if v is None:
            return "(empty)"
        if isinstance(v, str):
            s = v.strip()
            return s if s else "(empty)"
        try:
            if isinstance(v, float) and math.isnan(v):  # type: ignore[name-defined]
                return "(empty)"
        except Exception:
            pass
        s = str(v).strip()
        return s if s else "(empty)"

    tree_md = (tree_markdown_labels_only or "").strip()
    lab = _clean_text(item.get("label"))
    nam = _clean_text(item.get("name"))
    desc = _clean_text(item.get("description"))

    template = """\
        {role_prefix}system{role_suffix}
        You are an expert biomedical taxonomy matcher.

        ## TASK
        • From the TREE, choose **exactly one** label that best matches the ITEM.
        • Labels may include a short summary.
          Those summaries are guidance only; **output must be the exact label text (no summary)**.
        • If uncertain between close options, **prefer the closest correct parent** over a sibling.
        • Do **not** invent new labels; choose from the TREE only.
        • Candidates were preselected based on similarity to ITEM, they are not exhaustive (the Taxonomy is).

        The TREE is a nested (indented) Markdown list. Each bullet is:
        - <label> (<optional short summary>)

        ## OUTPUT (single-line JSON)
        {{"node_label":"...","rationale":"(≤ 20 words)"}}

        {role_prefix}user{role_suffix}
        ## TREE
        {TREE}.{eot}

        ## ITEM:
        - label: {LAB}
        - name: {NAM}
        - description: {DESC}{eot}
        {role_prefix}assistant{role_suffix}

    """
    return (
        dedent(template)
        .format(
            role_prefix=role_prefix,
            role_suffix=role_suffix,
            eot=eot,
            TREE=tree_md,
            LAB=lab,
            NAM=nam,
            DESC=desc,
        )
        .strip()
    )
