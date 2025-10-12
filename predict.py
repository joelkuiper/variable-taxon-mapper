from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional
from tqdm.auto import tqdm

from config import load_config
from main import run_pipeline
from src.evaluate import ProgressHook


def _parse_limit(value: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError("Row limit override must be an integer or 'none'.")
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return 0
    try:
        parsed = int(text)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError("Row limit override must be an integer or 'none'.") from exc
    if parsed < 0:
        raise ValueError("Row limit override must be non-negative or 'none'.")
    return parsed


def _make_tqdm_progress() -> ProgressHook:
    bar: Optional[Any] = None
    last_total = 0

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predictions for a variables-like CSV without evaluation.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the TOML configuration file controlling the run.",
    )
    parser.add_argument(
        "--variables",
        type=Path,
        default=None,
        help="Optional path to a variables-like CSV. Defaults to the config setting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to evaluation.results_csv in the config.",
    )
    parser.add_argument(
        "--limit",
        dest="row_limit_override",
        metavar="VALUE",
        default=argparse.SUPPRESS,
        help="Limit how many rows to predict (integer) or use 'none' for no limit.",
    )

    args = parser.parse_args(argv)

    if "row_limit_override" in vars(args):
        try:
            args.row_limit_override = _parse_limit(args.row_limit_override)
        except ValueError as exc:
            parser.error(str(exc))

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    config_path = args.config.resolve()
    base_path = config_path.parent
    config = load_config(config_path)

    if "row_limit_override" in vars(args):
        config.evaluation.n = args.row_limit_override

    variables_default, _ = config.data.to_paths(base_path)
    variables_path = (
        args.variables.resolve() if args.variables is not None else variables_default
    )

    progress_hook = _make_tqdm_progress()
    df, _ = run_pipeline(
        config,
        base_path=base_path,
        variables_csv=variables_path,
        evaluate=False,
        progress_hook=progress_hook,
    )

    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = (base_path / output_path).resolve()
    else:
        output_path = config.evaluation.resolve_results_path(
            base_path=base_path,
            variables_path=variables_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} predictions to {output_path}")


if __name__ == "__main__":
    main()
