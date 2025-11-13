"""Typer command for pruning configuration optimisation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from vtm.optimize_pruning import run_optimization
from vtm.utils import set_global_seed

from .app import app, logger
from .common import ConfigArgument, RowLimitOption, load_app_config


class PrunerChoice(str, Enum):
    NONE = "none"
    MEDIAN = "median"
    HALVING = "halving"


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
    trials: int = typer.Option(60, "--trials", min=1, help="Number of Optuna trials to evaluate."),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Optional random seed applied to Python, NumPy, and PyTorch.",
    ),
    storage: Optional[str] = typer.Option(
        None,
        "--storage",
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
        "--min-coverage",
        help="Minimum allowed_subtree_contains_gold_or_parent_rate to target.",
        show_default=True,
    ),
    min_possible: Optional[float] = typer.Option(
        None,
        "--min-possible",
        help="Minimum possible_correct_under_allowed_rate; defaults to --min-coverage when omitted.",
    ),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        help="Optional global timeout in seconds for study.optimize.",
    ),
    pruner: PrunerChoice = typer.Option(
        PrunerChoice.MEDIAN,
        "--pruner",
        help="Optuna pruner to use.",
        show_default=PrunerChoice.MEDIAN.value,
        case_sensitive=False,
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
        min=0,
        help="Enqueue this many seed trials for each pruning_mode to guarantee exploration.",
        show_default=True,
    ),
    tpe_startup: Optional[int] = typer.Option(
        None,
        "--tpe-startup",
        help="Override TPE n_startup_trials (random warmup). Default is max(24, 3 * repeats * num_modes).",
    ),
    tpe_multivariate: bool = typer.Option(
        False,
        "--tpe-multivariate",
        help="Enable TPE multivariate sampling (experimental; may warn).",
        show_default=True,
    ),
    tpe_constant_liar: bool = typer.Option(
        False,
        "--tpe-constant-liar",
        help="Enable TPE constant-liar (experimental; may warn).",
        show_default=True,
    ),
    suppress_experimental_warnings: bool = typer.Option(
        False,
        "--suppress-experimental-warnings",
        help="Suppress Optuna ExperimentalWarning messages.",
        show_default=True,
    ),
    enqueue_best_guess: bool = typer.Option(
        True,
        "--enqueue-best-guess/--no-enqueue-best-guess",
        help="Seed the study with the current config values when possible.",
        show_default=True,
    ),
) -> None:
    """Tune pruning parameters to maximise coverage and pruning quality."""

    config_path = config.resolve()
    base_path = config_path.parent
    config_obj = load_app_config(config_path)
    logger.info("Loaded configuration from %s", config_path)

    if seed is not None:
        set_global_seed(seed)

    run_optimization(
        config_obj,
        base_path=base_path,
        variables=variables,
        keywords=keywords,
        trials=trials,
        seed=seed,
        storage=storage,
        study_name=study_name,
        row_limit=row_limit,
        min_coverage=min_coverage,
        min_possible=min_possible,
        timeout=timeout,
        pruner=pruner.value,
        save_trials_csv=save_trials_csv,
        ensure_mode_repeats=ensure_mode_repeats,
        tpe_startup=tpe_startup,
        tpe_multivariate=tpe_multivariate,
        tpe_constant_liar=tpe_constant_liar,
        suppress_experimental_warnings=suppress_experimental_warnings,
        enqueue_best_guess=enqueue_best_guess,
    )
