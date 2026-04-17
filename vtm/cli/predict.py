from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer

from vtm.evaluate import ProgressHook
from vtm.pipeline import VariableTaxonMapper
from vtm.utils import ensure_file_exists, load_table, resolve_path, set_global_seed

from .app import app, logger
from .common import ConfigArgument, RowLimitOption, load_app_config
from ._output import (
    OutputFormat,
    build_sidecar_path,
    normalise_output_path,
    write_output,
)
from ._metadata import build_run_metadata


def _make_tqdm_progress() -> ProgressHook:
    bar: Optional[Any] = None
    last_total = 0

    from tqdm.auto import tqdm

    def _hook(done: int, total: int, _correct: Optional[int], _elapsed: float) -> None:
        nonlocal bar, last_total
        if bar is None:
            bar = tqdm(total=total, desc="Predicting", unit="item")
            last_total = total
        elif total != last_total:
            bar.total = total
            last_total = total

        if bar is not None:
            bar.n = done
            bar.refresh()
            if done >= total:
                bar.close()
                bar = None

    return _hook


@app.command("predict")
def predict_command(
    config: Path = ConfigArgument,
    variables: Optional[Path] = typer.Option(
        None,
        "--variables",
        "-v",
        help=(
            "Optional path to a variables-like table (CSV, Parquet, Feather). "
            "Defaults to the config setting."
        ),
        path_type=Path,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Optional output path. Defaults to evaluation.results_csv in the configuration. "
            "A JSON manifest is stored alongside the main output."
        ),
        path_type=Path,
    ),
    keywords: Optional[Path] = typer.Option(
        None,
        "--keywords",
        "-k",
        help=(
            "Optional path to the taxonomy keywords table (CSV, Parquet, Feather). "
            "Defaults to the config setting."
        ),
        path_type=Path,
    ),
    output_format: OutputFormat = typer.Option(
        "csv",
        "--output-format",
        "-f",
        help="File format for the generated predictions (csv, parquet, feather, json).",
        case_sensitive=False,
    ),
    compression: Optional[str] = typer.Option(
        None,
        "--compression",
        help="Optional compression codec forwarded to the pandas writer.",
        metavar="NAME",
    ),
    row_limit: Optional[int] = RowLimitOption,
) -> None:
    """Generate predictions for a variables-like table without evaluation.

    In addition to the primary output file, the command records a JSON run manifest that
    captures configuration, git metadata, and the output schema for FAIR reuse.
    """

    config_path = config.resolve()
    base_path = config_path.parent
    config_obj = load_app_config(config_path)
    logger.info("Loaded configuration from %s", config_path)

    set_global_seed(config_obj.seed)

    if row_limit is not None:
        config_obj.evaluation.n = row_limit
        logger.info("Row limit overridden to %s", row_limit)

    variables_default, keywords_default = config_obj.data.to_paths(base_path)
    variables_path = resolve_path(base_path, variables_default, variables)
    keywords_path = resolve_path(base_path, keywords_default, keywords)

    ensure_file_exists(variables_path, "variables data file")
    ensure_file_exists(keywords_path, "keywords data file")
    variables_df = load_table(variables_path, low_memory=False)
    logger.info(
        "Loaded variables frame with %d rows and %d columns",
        len(variables_df),
        len(variables_df.columns),
    )

    mapper = VariableTaxonMapper.from_config(
        config_obj,
        base_path=base_path,
        keywords_path=keywords_path,
    )
    progress_hook = _make_tqdm_progress()
    try:
        df, _ = mapper.predict(
            variables_df,
            evaluate=False,
            progress_hook=progress_hook,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "collect_predictions detected an active event loop" in message:
            logger.error(
                "Detected an active asyncio event loop. Run the CLI from a synchronous "
                "context or await vtm.evaluate.async_collect_predictions in your own service."
            )
            raise typer.Exit(code=1) from exc
        raise

    if output is not None:
        output_path = output if output.is_absolute() else (base_path / output).resolve()
    else:
        output_path = config_obj.evaluation.resolve_results_path(
            base_path=base_path,
            variables_path=variables_path,
        )

    output_path = normalise_output_path(output_path, output_format, compression)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output(
        df,
        output_path,
        output_format,
        compression=compression,
        keyword_columns=["resolved_keywords"],
        output_column_prefix=config_obj.evaluation.output_column_prefix,
    )
    logger.info(
        "Saved %d predictions to %s (%s)", len(df), output_path, output_format.upper()
    )

    manifest_path = build_sidecar_path(output_path, "manifest")
    manifest = {
        "_run_metadata": build_run_metadata(
            config=config_obj,
            config_path=config_path,
            base_path=base_path,
            variables_path=variables_path,
            keywords_path=keywords_path,
        ),
        "column_schema": [
            {"name": column, "dtype": str(dtype)}
            for column, dtype in zip(df.columns, df.dtypes)
        ],
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    logger.info("Saved manifest to %s", manifest_path)
