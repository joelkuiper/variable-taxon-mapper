from __future__ import annotations

from .app import app

# Import command modules so they register with the shared Typer application.
from . import predict as _predict  # noqa: F401
from . import prune_check as _prune_check  # noqa: F401
from . import evaluate as _evaluate  # noqa: F401
from . import summarize as _summarize  # noqa: F401
from . import optimize as _optimize  # noqa: F401

__all__ = ["app"]
