"""Prompt building and llama.cpp HTTP client helpers."""

from __future__ import annotations

import math
from textwrap import dedent
from typing import Any, Dict, Iterable, Optional, Union

import requests


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


_HTTP_SESSION = requests.Session()


def llama_completion(
    prompt: Union[str, Iterable[Union[str, int]]],
    endpoint: str,
    *,
    timeout: float = 120.0,
    session: Optional[requests.Session] = None,
    **kwargs: Any,
) -> str:
    payload: Dict[str, Any] = {"prompt": prompt}
    payload.update(kwargs)

    s = session or _HTTP_SESSION
    resp = s.post(endpoint, json=payload, timeout=(10, timeout))
    if not resp.ok:
        raise LlamaHTTPError(resp.status_code, resp.text)
    return resp.json().get("content", "")


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
