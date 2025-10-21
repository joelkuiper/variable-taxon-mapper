"""Helpers for evaluation parallelism."""
from __future__ import annotations

from config import HttpConfig, LLMConfig


def sock_read_timeout(http_cfg: HttpConfig, llm_cfg: LLMConfig) -> float:
    base = float(http_cfg.sock_read_floor)
    return max(base, float(llm_cfg.n_predict))
