"""Generate agreement reports for multiple error review CSV exports."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from tabulate import tabulate

NORMALIZED_DECISIONS = {"Accept", "Reject", "Unknown"}
_DECISION_MAP = {
    "a": "Accept",
    "accept": "Accept",
    "accepted": "Accept",
    "x": "Reject",
    "reject": "Reject",
    "rejected": "Reject",
    "u": "Unknown",
    "unknown": "Unknown",
}

_ID_CANDIDATES = ["row_index", "id", "item_id", "record_id"]
_DECISION_CANDIDATES = ["decision", "label", "verdict"]


class AgreementReportError(RuntimeError):
    """Custom error to signal invalid input data."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an agreement report for multiple error_review_cli CSV exports."
        )
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="Two or more CSV files to compare (one per rater).",
    )
    parser.add_argument(
        "--id-column",
        dest="id_column",
        help=(
            "Name of the item identifier column. If omitted, the script will try to "
            "detect a shared column among the inputs."
        ),
    )
    parser.add_argument(
        "--decision-column",
        dest="decision_column",
        help=(
            "Name of the decision column. If omitted, the script will try to detect "
            "a shared column among the inputs."
        ),
    )
    parser.add_argument(
        "--rater-names",
        nargs="*",
        help=(
            "Optional explicit rater names. Provide the same number as CSV files; "
            "defaults to the CSV file stem."
        ),
    )
    return parser.parse_args(argv)


def _infer_common_column(
    frames: Sequence[pd.DataFrame],
    requested: str | None,
    candidates: Sequence[str],
    *,
    description: str,
) -> str:
    if requested:
        lowered = requested.lower()
        for idx, frame in enumerate(frames):
            if lowered not in {col.lower() for col in frame.columns}:
                raise AgreementReportError(
                    f"{description} '{requested}' not found in CSV #{idx + 1}."
                )
        return lowered

    lower_columns = [
        {column.lower(): column for column in frame.columns} for frame in frames
    ]

    for candidate in candidates:
        lowered = candidate.lower()
        if all(lowered in mapping for mapping in lower_columns):
            return lowered

    shared = set(lower_columns[0]) if lower_columns else set()
    for mapping in lower_columns[1:]:
        shared &= set(mapping)
    if shared:
        return sorted(shared)[0]

    raise AgreementReportError(
        f"Could not determine a shared {description.lower()} column among the inputs."
    )


def _extract_columns(
    frame: pd.DataFrame,
    id_lower: str,
    decision_lower: str,
    rater_name: str,
) -> pd.DataFrame:
    mapping = {column.lower(): column for column in frame.columns}
    try:
        id_column = mapping[id_lower]
        decision_column = mapping[decision_lower]
    except KeyError as exc:  # pragma: no cover - guarded by detection step
        raise AgreementReportError(f"Required column missing: {exc.args[0]}") from exc

    subset = frame[[id_column, decision_column]].copy()
    subset.rename(columns={id_column: "item_id", decision_column: rater_name}, inplace=True)
    subset.drop_duplicates(subset="item_id", inplace=True)
    return subset


def _normalize_decision(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise AgreementReportError("Encountered empty decision value during normalization.")
    if normalized in _DECISION_MAP:
        return _DECISION_MAP[normalized]
    raise AgreementReportError(f"Unsupported decision value: {value!r}")


def _normalize_frame(frame: pd.DataFrame, rater_columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in rater_columns:
        normalized[column] = normalized[column].map(_normalize_decision)
    return normalized


def _compute_pairwise_agreements(
    frame: pd.DataFrame, rater_columns: Sequence[str]
) -> list[dict[str, object]]:
    total = len(frame)
    if total == 0:
        return []

    rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(rater_columns, 2):
        agreements = (frame[left] == frame[right]).sum()
        rows.append(
            {
                "Rater A": left,
                "Rater B": right,
                "Agreement": f"{agreements / total * 100:.2f}%",
                "Matching Items": agreements,
            }
        )
    return rows


def _compute_per_label_agreement(
    frame: pd.DataFrame, rater_a: str, rater_b: str
) -> list[dict[str, object]]:
    total = len(frame)
    if total == 0:
        return []
    rows: list[dict[str, object]] = []
    for label in sorted(NORMALIZED_DECISIONS):
        matches = ((frame[rater_a] == label) & (frame[rater_b] == label)).sum()
        rows.append(
            {
                "Label": label,
                "Matching Items": matches,
                "Percent": f"{matches / total * 100:.2f}%",
            }
        )
    return rows


def _fleiss_kappa(frame: pd.DataFrame, rater_columns: Sequence[str]) -> float:
    n_items = len(frame)
    n_raters = len(rater_columns)
    if n_items == 0 or n_raters < 2:
        return float("nan")

    label_list = sorted(NORMALIZED_DECISIONS)
    counts = pd.DataFrame(0, index=frame.index, columns=label_list)
    for column in rater_columns:
        for label in label_list:
            counts[label] += (frame[column] == label).astype(int)

    n_matrix = counts.to_numpy(dtype=float)
    p_i = (n_matrix * (n_matrix - 1)).sum(axis=1) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_category = n_matrix.sum(axis=0) / (n_items * n_raters)
    p_e = (p_category**2).sum()
    if p_e == 1:
        return 1.0
    return float((p_bar - p_e) / (1 - p_e))


def _compute_consensus_stats(
    frame: pd.DataFrame, rater_columns: Sequence[str]
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    total = len(frame)
    unanimous_mask = frame[rater_columns].nunique(axis=1) == 1
    unanimous_count = int(unanimous_mask.sum())

    label_counts = (
        frame.loc[unanimous_mask, rater_columns[0]]
        .value_counts()
        .reindex(sorted(NORMALIZED_DECISIONS), fill_value=0)
    )

    strict_majority_mask = frame[rater_columns].apply(
        lambda row: row.value_counts().max() > len(rater_columns) / 2,
        axis=1,
    ).astype(bool)
    no_majority_ids = frame.loc[~strict_majority_mask, "item_id"].tolist()
    no_majority_count = len(no_majority_ids)

    summary = {
        "Unanimous Items": unanimous_count,
        "Unanimous %": f"{(unanimous_count / total * 100) if total else 0:.2f}%",
        "Items Without Majority": no_majority_count,
        "No Majority %": f"{(no_majority_count / total * 100) if total else 0:.2f}%",
    }

    unanimous_table = pd.DataFrame(
        {
            "Label": label_counts.index,
            "Unanimous Items": label_counts.values,
            "Percent": [
                f"{(count / total * 100) if total else 0:.2f}%" for count in label_counts.values
            ],
        }
    )

    no_majority_table = frame.loc[~strict_majority_mask, ["item_id", *rater_columns]].copy()
    if len(no_majority_table) > 20:
        no_majority_table = no_majority_table.head(20)
    return summary, unanimous_table, no_majority_table


def _format_table(data: Iterable[dict[str, object]] | pd.DataFrame) -> str:
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return "(no data)"
        records = data.to_dict(orient="records")
        return tabulate(records, headers="keys", tablefmt="rounded_grid", showindex=False)
    data = list(data)
    if not data:
        return "(no data)"
    return tabulate(data, headers="keys", tablefmt="rounded_grid", showindex=False)


def generate_report(args: argparse.Namespace) -> None:
    if len(args.csv_paths) < 2:
        raise AgreementReportError("Please provide at least two CSV files.")

    frames = [pd.read_csv(path, dtype=str).fillna("") for path in args.csv_paths]
    id_lower = _infer_common_column(
        frames,
        args.id_column,
        _ID_CANDIDATES,
        description="ID column",
    )
    decision_lower = _infer_common_column(
        frames,
        args.decision_column,
        _DECISION_CANDIDATES,
        description="Decision column",
    )

    if args.rater_names:
        if len(args.rater_names) != len(args.csv_paths):
            raise AgreementReportError("Number of rater names must match number of CSV files.")
        rater_names = list(args.rater_names)
    else:
        rater_names = [path.stem for path in args.csv_paths]

    extracted = [
        _extract_columns(frame, id_lower, decision_lower, rater_name)
        for frame, rater_name in zip(frames, rater_names)
    ]

    merged = extracted[0]
    for additional in extracted[1:]:
        merged = merged.merge(additional, on="item_id", how="inner")

    merged = _normalize_frame(merged, rater_names)

    total_items = len(merged)
    n_raters = len(rater_names)
    summary_rows = [
        {"Metric": "Total Items", "Value": total_items},
        {"Metric": "Raters", "Value": n_raters},
    ]

    overall: float | None = None
    kappa: float | None = None

    if total_items == 0:
        metric_label = "Percent Agreement" if n_raters == 2 else "Fleiss' Kappa"
        summary_rows.append({"Metric": metric_label, "Value": "n/a"})
        print("Agreement Report")
        print("=" * 17)
        print(_format_table(summary_rows))
        print()
        print("No overlapping items across raters; nothing to report.")
        return

    if n_raters == 2:
        r1, r2 = rater_names
        overall_value = float((merged[r1] == merged[r2]).mean())
        overall = overall_value
        summary_rows.append({"Metric": "Percent Agreement", "Value": f"{overall_value * 100:.2f}%"})
    else:
        kappa = _fleiss_kappa(merged, rater_names)
        summary_rows.append({"Metric": "Fleiss' Kappa", "Value": f"{kappa:.4f}"})

    print("Agreement Report")
    print("=" * 17)
    print(_format_table(summary_rows))
    print()

    pairwise_rows = _compute_pairwise_agreements(merged, rater_names)
    print("Pairwise Agreement")
    print("------------------")
    print(_format_table(pairwise_rows))
    print()

    if n_raters == 2:
        r1, r2 = rater_names
        assert overall is not None
        print("Overall Percent Agreement")
        print("---------------------------")
        print(_format_table([{"Agreement": f"{overall * 100:.2f}%"}]))
        print()

        label_rows = _compute_per_label_agreement(merged, r1, r2)
        print("Agreement by Label")
        print("-------------------")
        print(_format_table(label_rows))

        disagreement_table = merged.loc[merged[r1] != merged[r2], ["item_id", r1, r2]]
        if len(disagreement_table) > 20:
            disagreement_table = disagreement_table.head(20)
        if not disagreement_table.empty:
            print()
            print("Items Needing Re-review")
            print("-----------------------")
            print(_format_table(disagreement_table))
        return

    # n_raters >= 3
    assert kappa is not None
    print("Fleiss' Kappa")
    print("--------------")
    print(_format_table([{"Kappa": f"{kappa:.4f}"}]))
    print()

    summary, unanimous_table, no_majority_table = _compute_consensus_stats(
        merged, rater_names
    )
    print("Consensus Summary")
    print("-----------------")
    summary_rows = [{"Metric": key, "Value": value} for key, value in summary.items()]
    print(_format_table(summary_rows))
    print()

    print("Unanimous Decisions by Label")
    print("-----------------------------")
    print(_format_table(unanimous_table))

    if not no_majority_table.empty:
        print()
        print("Items Needing Re-review")
        print("-----------------------")
        print(_format_table(no_majority_table))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        generate_report(args)
    except AgreementReportError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()
