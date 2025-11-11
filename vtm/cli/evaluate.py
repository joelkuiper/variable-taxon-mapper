from __future__ import annotations

import json
from pathlib import Path

import typer

from vtm.pipeline import VariableTaxonMapper
from vtm.reporting import report_results
from vtm.utils import ensure_file_exists, load_table, set_global_seed

from .app import app, logger
from .common import ConfigArgument, load_app_config


@app.command("evaluate")
def run_command(config: Path = ConfigArgument) -> None:
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

    metrics_path = results_path.with_name(f"{results_path.stem}_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    logger.info("Metrics saved to %s", metrics_path)

    report_results(df, metrics)
