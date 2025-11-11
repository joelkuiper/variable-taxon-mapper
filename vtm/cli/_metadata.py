"""Shared helpers for assembling run metadata manifests."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import subprocess
from pathlib import Path
from typing import Any


def _to_serialisable(value: Any) -> Any:
    """Convert ``value`` into JSON-serialisable types."""

    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(key): _to_serialisable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_serialisable(item) for item in value]

    return value


def _redact_api_key(data: MutableMapping[str, Any]) -> None:
    """Ensure sensitive authentication tokens are not persisted."""

    if not isinstance(data, MutableMapping):
        return
    api_key = data.get("api_key")
    if api_key:
        data["api_key"] = "***redacted***"


def detect_git_commit(base_path: Path) -> str | None:
    """Return the current git commit SHA for ``base_path`` when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=base_path,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def compute_sha256(path: Path) -> str | None:
    """Compute the SHA256 digest for ``path`` if the file exists."""

    try:
        handle = path.open("rb")
    except FileNotFoundError:
        return None

    digest = hashlib.sha256()
    with handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_metadata(
    *,
    config: Any,
    config_path: Path,
    base_path: Path,
    variables_path: Path,
    keywords_path: Path,
) -> dict[str, Any]:
    """Collect metadata describing the current run for provenance."""

    config_dict = _to_serialisable(config)

    llm_params: Any = None
    if isinstance(config_dict, Mapping):
        llm_section = config_dict.get("llm")
        if isinstance(llm_section, MutableMapping):
            _redact_api_key(llm_section)
            llm_params = dict(llm_section)
        elif isinstance(llm_section, Mapping):
            llm_params = dict(llm_section)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "git_commit": detect_git_commit(base_path),
        },
        "config": {
            "path": str(config_path),
            "sha256": compute_sha256(config_path),
            "data": config_dict,
        },
        "inputs": {
            "variables": {
                "path": str(variables_path),
                "sha256": compute_sha256(variables_path),
            },
            "keywords": {
                "path": str(keywords_path),
                "sha256": compute_sha256(keywords_path),
            },
        },
        "llm_parameters": llm_params,
    }

    return metadata


__all__ = [
    "build_run_metadata",
    "compute_sha256",
    "detect_git_commit",
]

