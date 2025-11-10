from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from vtm.config import AppConfig, FieldMappingConfig, coerce_config, load_config


def _row_limit_callback(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return 0
    try:
        parsed = int(text)
    except ValueError as exc:
        raise typer.BadParameter("Row limit must be an integer or 'none'.") from exc
    if parsed < 0:
        raise typer.BadParameter("Row limit must be non-negative or 'none'.")
    return parsed


ConfigArgument = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    path_type=Path,
    help="Path to the TOML configuration file controlling the run.",
)

RowLimitOption = typer.Option(
    None,
    "--limit",
    callback=_row_limit_callback,
    help="Limit how many rows to process (integer) or use 'none' for no limit.",
)


def load_app_config(config_path: Path) -> AppConfig:
    config_path = config_path.resolve()
    return load_config(config_path)


__all__ = [
    "AppConfig",
    "ConfigArgument",
    "FieldMappingConfig",
    "RowLimitOption",
    "coerce_config",
    "load_app_config",
]
