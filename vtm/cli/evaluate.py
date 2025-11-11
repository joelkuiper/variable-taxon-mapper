from __future__ import annotations

import json
from pathlib import Path

import typer

from vtm.pipeline import VariableTaxonMapper
from vtm.reporting import report_results
from vtm.utils import ensure_file_exists, load_table, set_global_seed

from .app import app, logger
from .common import ConfigArgument, load_app_config
from ._metadata import build_run_metadata

@app.command("evaluate")
def run_command(
    config: Path = ConfigArgument,
    summary_md: Path | None = typer.Option(
        None,
        "--summary-md",
        help="Write the evaluation summary to the given Markdown file.",
        metavar="PATH",
    ),
    summary_text: Path | None = typer.Option(
        None,
        "--summary-text",
        help="Also persist a plain-text summary to the given path.",
        metavar="PATH",
    ),
) -> None:
    """Run the full variable taxonomy mapping pipeline with evaluation."""

    config_path = config.resolve()
    base_path = config_path.parent
    config_obj = load_app_config(config_path)
    logger.info("Loaded configuration from %s", config_path)

    set_global_seed(config_obj.seed)

    variables_path, keywords_path = config_obj.data.to_paths(base_path)
    ensure_file_exists(variables_path, "variables data file")

    mapper = VariableTaxonMapper.from_config(
        config_obj,
        base_path=base_path,
        keywords_path=keywords_path,
    )

    variables = load_table(variables_path, low_memory=False)
    logger.info(
        "Loaded variables frame with %d rows and %d columns",
        len(variables),
        len(variables.columns),
    )

    try:
        df, metrics = mapper.predict(variables)
    except RuntimeError as exc:
        message = str(exc)
        if "collect_predictions detected an active event loop" in message:
            logger.error(
                "Detected an active asyncio event loop. Run the mapper from a synchronous "
                "context or await vtm.evaluate.async_collect_predictions inside your service."
            )
            raise typer.Exit(code=1) from exc
        raise

    results_path = config_obj.evaluation.resolve_results_path(
        base_path=base_path,
        variables_path=variables_path,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info("Results saved to %s", results_path)

    run_metadata = build_run_metadata(
        config=config_obj,
        config_path=config_path,
        base_path=base_path,
        variables_path=variables_path,
        keywords_path=keywords_path,
    )

    # Metrics + metadata
    metrics_path = results_path.with_name(f"{results_path.stem}_metrics.json")
    metrics_payload = dict(metrics)
    metrics_payload["_run_metadata"] = run_metadata
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, sort_keys=True)
    logger.info("Metrics saved to %s", metrics_path)

    manifest_path = results_path.with_name(f"{results_path.stem}_manifest.json")
    manifest = {
        "_run_metadata": run_metadata,
        "column_schema": [
            {"name": column, "dtype": str(dtype)}
            for column, dtype in zip(df.columns, df.dtypes)
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    logger.info("Manifest saved to %s", manifest_path)

    summary_md = summary_md.resolve() if summary_md else None
    summary_text = summary_text.resolve() if summary_text else None

    if summary_text is not None and summary_md is None:
        logger.error("--summary-text requires --summary-md to also be set")
        raise typer.Exit(code=1)

    output_path: Path | tuple[Path, Path] | None
    if summary_md and summary_text:
        output_path = (summary_md, summary_text)
    elif summary_md:
        output_path = summary_md
    else:
        output_path = None

    field_cfg = config_obj.fields
    metadata_columns = field_cfg.metadata_columns_list()
    display_columns = list(dict.fromkeys(metadata_columns))
    gold_column = field_cfg.gold_column()
    if gold_column and gold_column not in display_columns:
        display_columns.append(gold_column)
    for extra in ["gold_labels", "resolved_label", "correct", "match_type"]:
        if extra not in display_columns:
            display_columns.append(extra)

    dataset_column = field_cfg.dataset_column_name() or "dataset"

    report_results(
        df,
        metrics,
        display_columns=display_columns,
        dataset_column=dataset_column,
        output_path=output_path,
    )
