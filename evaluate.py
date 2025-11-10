from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from vtm.config import AppConfig
from vtm.evaluate import ProgressHook
from vtm.pipeline import VariableTaxonMapper
from vtm.utils import ensure_file_exists, resolve_path, set_global_seed
from vtm.cli import app as cli_app


try:
    from typer import Exit as _TyperExit
    from typer.main import get_command as _get_typer_command
except ImportError:  # pragma: no cover - typer always available via project deps
    _TyperExit = None  # type: ignore[assignment]
    _get_typer_command = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def run_pipeline(
    config: AppConfig,
    *,
    base_path: Path | None = None,
    variables_csv: Path | None = None,
    evaluate: bool = True,
    progress_hook: ProgressHook | None = None,
) -> Tuple[pd.DataFrame, dict[str, object]]:
    logger.debug("Setting global seed to %s", config.seed)
    set_global_seed(config.seed)

    variables_default, keywords_default = config.data.to_paths(base_path)

    variables_path = resolve_path(base_path, variables_default, variables_csv)
    keywords_path = resolve_path(base_path, keywords_default, None)

    ensure_file_exists(variables_path, "variables CSV")

    parallel_cfg = config.parallelism
    logger.info(
        "[pipeline] concurrency settings: pruning_workers=%s, pruning_batch=%s",
        parallel_cfg.pruning_workers,
        parallel_cfg.pruning_batch_size,
    )

    variables = pd.read_csv(variables_path, low_memory=False)
    logger.info(
        "Loaded variables frame with %d rows and %d columns", len(variables), len(variables.columns)
    )
    logger.debug("Resolved variables path: %s", variables_path)
    logger.debug("Resolved keywords path: %s", keywords_path)

    mapper = VariableTaxonMapper.from_config(
        config,
        base_path=base_path,
        keywords_path=keywords_path,
    )

    df, metrics = mapper.predict(
        variables,
        evaluate=evaluate,
        progress_hook=progress_hook,
    )

    return df, metrics


def main(argv: list[str] | None = None) -> None:
    """Delegate to the unified Typer CLI when executed as a script."""

    if _get_typer_command is None or _TyperExit is None:  # pragma: no cover - safety
        raise RuntimeError("Typer is required to invoke the unified CLI")

    if argv is None:
        import sys

        args = list(sys.argv[1:])
    else:
        args = list(argv)

    command = _get_typer_command(cli_app)
    try:
        command.main(
            args=["run", *args],
            prog_name="vtm",
            standalone_mode=False,
        )
    except _TyperExit as exc:  # pragma: no cover - pass through exit code
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
