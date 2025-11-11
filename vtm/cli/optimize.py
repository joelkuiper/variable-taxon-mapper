from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from optimize_pruning import run_optimization

from vtm.utils import set_global_seed

from .app import app, logger
from .common import ConfigArgument, RowLimitOption, load_app_config


@app.command("optimize-pruning")
def optimize_pruning_command(
    config: Path = ConfigArgument,
    variables: Optional[Path] = typer.Option(
        None,
        "--variables",
        help="Optional override for the variables CSV file.",
        path_type=Path,
    ),
    keywords: Optional[Path] = typer.Option(
        None,
        "--keywords",
        help="Optional override for the taxonomy keywords CSV file.",
        path_type=Path,
    ),
    trials: int = typer.Option(60, help="Number of Optuna trials to evaluate."),
    seed: Optional[int] = typer.Option(
        None,
        help="Optional random seed overriding the configuration.",
    ),
    storage: Optional[str] = typer.Option(
        None,
        help="Optional Optuna storage URL for persisting studies.",
    ),
    study_name: Optional[str] = typer.Option(
        None,
        "--study-name",
        help="Optional Optuna study name when using persistent storage.",
    ),
    row_limit: Optional[int] = RowLimitOption,
    min_coverage: float = typer.Option(
        0.97,
        help="Minimum allowed_subtree_contains_gold_or_parent_rate to target.",
    ),
    min_possible: Optional[float] = typer.Option(
        None,
        help="Minimum possible_correct_under_allowed_rate to accept.",
    ),
    timeout: Optional[int] = typer.Option(
        None,
        help="Optional global timeout in seconds for study.optimize.",
    ),
    pruner: str = typer.Option(
        "median",
        help="Optuna pruner to use (choices: none, median, halving).",
    ),
    save_trials_csv: Optional[Path] = typer.Option(
        None,
        "--save-trials-csv",
        help="Optional path to dump all completed trial results as CSV.",
        path_type=Path,
    ),
    ensure_mode_repeats: int = typer.Option(
        2,
        "--ensure-mode-repeats",
        help="Enqueue this many seed trials for each pruning_mode to guarantee exploration.",
    ),
    tpe_startup: Optional[int] = typer.Option(
        None,
        "--tpe-startup",
        help="Override TPE n_startup_trials (random warmup).",
    ),
    tpe_multivariate: bool = typer.Option(
        False,
        "--tpe-multivariate",
        help="Enable TPE multivariate sampling (experimental; may warn).",
    ),
    tpe_constant_liar: bool = typer.Option(
        False,
        "--tpe-constant-liar",
        help="Enable TPE constant-liar (experimental; may warn).",
    ),
    suppress_experimental_warnings: bool = typer.Option(
        False,
        "--suppress-experimental-warnings",
        help="Suppress Optuna ExperimentalWarning messages.",
    ),
) -> None:
    """Optimize pruning configuration parameters using Optuna."""

    config_path = config.resolve()
    base_path = config_path.parent
    config_obj = load_app_config(config_path)
    logger.info("Loaded configuration from %s", config_path)

    effective_seed = seed if seed is not None else config_obj.seed
    set_global_seed(int(effective_seed))

    if save_trials_csv is not None and not save_trials_csv.is_absolute():
        save_trials_csv = (base_path / save_trials_csv).resolve()

    run_optimization(
        config_obj,
        base_path=base_path,
        variables=variables,
        keywords=keywords,
        trials=trials,
        seed=effective_seed,
        storage=storage,
        study_name=study_name,
        row_limit=row_limit,
        min_coverage=min_coverage,
        min_possible=min_possible,
        timeout=timeout,
        pruner_name=pruner,
        save_trials_csv=save_trials_csv,
        ensure_mode_repeats=ensure_mode_repeats,
        tpe_startup=tpe_startup,
        tpe_multivariate=tpe_multivariate,
        tpe_constant_liar=tpe_constant_liar,
        suppress_experimental_warnings=suppress_experimental_warnings,
    )

