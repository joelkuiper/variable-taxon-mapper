from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from config import load_config
from main import format_metrics, run_pipeline


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate taxonomy predictions for a CSV of variables."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the TOML configuration file controlling the run.",
    )
    parser.add_argument(
        "variables_csv",
        type=Path,
        help="CSV file containing the variables to label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. If omitted, defaults to the evaluation"
            " configuration's results_csv (or <input>_results.csv)."
        ),
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="If provided, print the evaluation metrics summary for reference.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    base_path = config_path.parent
    config = load_config(config_path)

    variables_csv = args.variables_csv.resolve()
    df, metrics = run_pipeline(
        config,
        base_path=base_path,
        variables_csv=variables_csv,
    )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else config.evaluation.resolve_results_path(
            base_path=base_path,
            variables_path=variables_csv,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    display_cols = [
        "dataset",
        "label",
        "name",
        "description",
        "resolved_label",
        "resolved_id",
        "resolved_path",
    ]
    present = [c for c in display_cols if c in df.columns]
    if present:
        print(df[present].head())
    else:
        print(df.head())

    print(f"Predictions saved to {output_path}")

    if args.show_metrics:
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()
