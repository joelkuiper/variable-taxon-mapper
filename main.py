from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd

from config import AppConfig, load_config
from vtm.evaluate import ProgressHook
from vtm.pipeline import VariableTaxonMapper
from vtm.utils import (
    configure_logging,
    ensure_file_exists,
    resolve_path,
    set_global_seed,
)
from vtm.reporting import report_results


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the variable taxonomy mapper")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the TOML configuration file controlling the run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging(level=os.getenv("LOG_LEVEL", logging.INFO))

    args = parse_args(argv)
    config_path = args.config.resolve()
    base_path = config_path.parent
    config = load_config(config_path)
    logger.info("Loaded configuration from %s", config_path)
    set_global_seed(config.seed)
    variables_path, keywords_path = config.data.to_paths(base_path)
    mapper = VariableTaxonMapper.from_config(
        config,
        base_path=base_path,
        keywords_path=keywords_path,
    )

    ensure_file_exists(variables_path, "variables CSV")
    variables = pd.read_csv(variables_path, low_memory=False)
    logger.info(
        "Loaded variables frame with %d rows and %d columns",
        len(variables),
        len(variables.columns),
    )

    df, metrics = mapper.predict(variables)

    results_path = config.evaluation.resolve_results_path(
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


if __name__ == "__main__":
    main()
