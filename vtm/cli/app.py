from __future__ import annotations

import logging
import os
from typing import Optional

import typer

from vtm.utils import configure_logging


LOGGER_NAME = "vtm.cli"


def _resolve_log_level(log_level: Optional[str]) -> int | str:
    if log_level:
        return log_level
    env_level = os.getenv("LOG_LEVEL")
    return env_level if env_level else logging.INFO


def create_app() -> typer.Typer:
    app = typer.Typer(help="Unified command line interface for Variable Taxon Mapper")

    @app.callback()
    def _configure_cli(
        log_level: Optional[str] = typer.Option(None, help="Python logging level"),
    ) -> None:
        """Configure logging before running any command."""

        configure_logging(level=_resolve_log_level(log_level))

    return app


logger = logging.getLogger(LOGGER_NAME)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = create_app()

__all__ = ["app", "logger"]
