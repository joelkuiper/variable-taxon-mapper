"""Prompt building and llama.cpp HTTP client helpers."""

from __future__ import annotations

import asyncio
import json
import os
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

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


_OPENAI_SYSTEM_PROMPT = dedent(
    """\
    You are a taxonomy matching assistant. Carefully read the TREE (and optional SUGGESTIONS) and then follow the task instructions.

    # TASK
    • Choose **exactly one** concept from the TREE or SUGGESTIONS that best matches the ITEM.
    • Prefer the most specific matching child; fall back to its parent only when no child fits.
    • The TREE is a nested (indented) Markdown list. Each bullet is formatted as `- <concept label> [<optional short description>]`.
    • Text inside square brackets is descriptive guidance only. **Return the exact concept label without the description.**
    • If nothing fits, return the fallback label provided in the TREE or SUGGESTIONS (often `NO MATCH`).

    # RESPONSE FORMAT
    Reply with a single-line JSON object exactly like `{"concept_label":"..."}` containing only the chosen concept label.
    Do not add any narration, Markdown, or additional keys.
    """
).strip()


def _build_user_content(
    tree_md: str,
    *,
    item_label: str,
    item_name: str,
    item_dataset: str,
    item_desc: str,
) -> str:
    sections: List[str] = []
    if tree_md:
        sections.append(tree_md)
        sections.append("")
    sections.extend(
        [
            "# ITEM:",
            f"**{item_label}** ({item_name})",
            f"dataset: {item_dataset}",
            f"description: {item_desc}",
        ]
    )
    return "\n".join(sections).strip()


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

            try:
                raw_body = await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                raise RuntimeError(
                    f"Failed to read response from LLM endpoint {endpoint!r}: {exc}"
                ) from exc

            if not raw_body.strip():
                raise RuntimeError(
                    "LLM endpoint returned an empty response body; "
                    "cannot extract completion."
                )

            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "LLM endpoint returned invalid JSON; cannot extract completion."
                ) from exc

            if "error" in data and data.get("error"):
                raise RuntimeError(
                    f"LLM endpoint reported an error: {data.get('error')}"
                )

            content = data.get("content")
            if isinstance(content, list):
                parts: List[str] = []
                for chunk in content:
                    if isinstance(chunk, dict):
                        text = chunk.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                    elif isinstance(chunk, str):
                        parts.append(chunk)
                content = "".join(parts)

            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("LLM endpoint response is missing completion text.")

            return content
    except asyncio.CancelledError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise RuntimeError(
            f"Error while contacting LLM endpoint {endpoint!r}: {exc}"
        ) from exc
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

        pending_tasks: Set[asyncio.Task[str]] = set(tasks)

        while pending_tasks:
            done, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_EXCEPTION
            )

            first_exc: Optional[BaseException] = None
            for task in done:
                exc = task.exception()
                if exc is not None:
                    first_exc = exc
                    break

            if first_exc is not None:
                for task in pending_tasks:
                    task.cancel()
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                raise first_exc

        results = [task.result() for task in tasks]
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
    style: str = "llamacpp",
    role_prefix: str = "<|im_start|>",
    role_suffix: str = "",
    eot: str = "<|im_end|>",
) -> Union[str, List[Dict[str, str]]]:
    tree_md = (tree_markdown_labels_only or "").strip()
    lab = clean_text(item.get("label"))
    nam = clean_text(item.get("name"))
    dat = clean_text(item.get("dataset"))
    desc = clean_text(item.get("description"))

    if style.lower() == "openai":
        system_text = _OPENAI_SYSTEM_PROMPT
        user_text = _build_user_content(
            tree_md,
            item_label=lab,
            item_name=nam,
            item_dataset=dat,
            item_desc=desc,
        )
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]

    template = """\
        {role_prefix}system{role_suffix}
        # TASK
        • From the TREE (or SUGGESTIONS), choose **exactly one** concept that best matches the ITEM.
        • Prefer the most specific matching child, if present; only choose the parent if no child fits.
        • The TREE is a nested (indented) Markdown list where each bullet is: `- <concept label> [<optional short description>]`.
        • Concepts may include a short description in square brackets. Those descriptions are guidance only; **output must be the exact concept label (no description)**.
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


async def openai_chat_completion_many(
    requests: Sequence[Dict[str, Any]],
    *,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    api_base: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    timeout: float = 120.0,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[str]:
    """Resolve Chat Completions via OpenAI's REST API."""

    if not requests:
        return []

    key = api_key
    if not key and api_key_env:
        key = os.environ.get(api_key_env)

    if not key:
        raise RuntimeError(
            "OpenAI API key missing; set llm.api_key or provide the environment variable"
            f" {api_key_env!r}."
        )

    base_url = (api_base or "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project

    close_session = False
    if session is None:
        timeout_cfg = aiohttp.ClientTimeout(
            total=None, sock_connect=10, sock_read=timeout
        )
        session = aiohttp.ClientSession(timeout=timeout_cfg)
        close_session = True

    async def _run_single(payload: Dict[str, Any]) -> str:
        request_payload = dict(payload)
        request_timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=10, sock_read=timeout
        )
        try:
            async with session.post(
                endpoint,
                json=request_payload,
                headers=headers,
                timeout=request_timeout,
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(
                        f"OpenAI API returned HTTP {resp.status}: {body}"
                    )

                try:
                    raw_body = await resp.text()
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    raise RuntimeError(
                        "Failed to read response from OpenAI API: {exc}".format(exc=exc)
                    ) from exc

        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise RuntimeError(
                f"Error while contacting OpenAI API at {endpoint!r}: {exc}"
            ) from exc

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI API returned invalid JSON; cannot extract completion."
            ) from exc

        error_payload = data.get("error")
        if error_payload:
            if isinstance(error_payload, dict):
                message = error_payload.get("message") or str(error_payload)
            else:
                message = str(error_payload)
            raise RuntimeError(f"OpenAI API reported an error: {message}")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI API response is missing completion choices.")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI API response is missing message content.")

        content = message.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text_part = part.get("text")
                    if isinstance(text_part, str):
                        parts.append(text_part)
                elif isinstance(part, str):
                    parts.append(part)
            content = "".join(parts)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI API response is missing completion text.")

        return content

    try:
        tasks = [asyncio.create_task(_run_single(payload)) for payload in requests]

        pending: Set[asyncio.Task[str]] = set(tasks)
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
        if close_session and session is not None:
            await session.close()
