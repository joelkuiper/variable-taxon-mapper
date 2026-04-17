"""Interactive adjudication tool for multi-rater error reviews."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from vtm.utils import load_table

# Candidate column names mirror the agreement_report script so both CLIs accept
# the same arguments and conventions.
_ID_CANDIDATES = ["row_index", "id", "item_id", "record_id"]
_DECISION_CANDIDATES = ["decision", "label", "verdict"]

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

ADJUDICATION_KEYS = {
    "a": "Accept",
    "x": "Reject",
    "u": "Unsure",
}


class AdjudicationError(RuntimeError):
    """Custom exception for adjudication specific validation errors."""


@dataclass
class AdjudicationItem:
    """Container for a single item that lacks rater consensus."""

    item_id: str
    metadata: dict[str, str]
    rater_decisions: dict[str, str]

    def to_output_row(
        self,
        final_decision: str,
        metadata_columns: Sequence[str],
        rater_names: Sequence[str],
    ) -> dict[str, str]:
        row: dict[str, str] = {"item_id": self.item_id}
        for column in metadata_columns:
            row[column] = self.metadata.get(column, "")
        for name in rater_names:
            row[f"{name}_decision"] = self.rater_decisions.get(name, "")
        row["final_decision"] = final_decision
        return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively adjudicate disagreements across multiple "
            "error_review_cli CSV exports."
        )
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="Two or more CSV files produced by error_review_cli (one per rater).",
    )
    parser.add_argument(
        "--id-column",
        dest="id_column",
        help=(
            "Name of the shared identifier column. Defaults to auto-detection "
            "based on common column names."
        ),
    )
    parser.add_argument(
        "--decision-column",
        dest="decision_column",
        help=(
            "Name of the decision column. Defaults to auto-detection based on "
            "common column names."
        ),
    )
    parser.add_argument(
        "--rater-names",
        nargs="*",
        help=(
            "Optional display names for the raters. Provide the same number as "
            "CSV files; defaults to the CSV file stem."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adjudication_decisions.csv"),
        help="Destination CSV for final adjudication verdicts.",
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
                raise AdjudicationError(
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

    raise AdjudicationError(
        f"Could not determine a shared {description.lower()} column among the inputs."
    )


def _normalize_decision(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise AdjudicationError("Encountered empty decision value during normalization.")
    if normalized in _DECISION_MAP:
        return _DECISION_MAP[normalized]
    raise AdjudicationError(f"Unsupported decision value: {value!r}")


def _load_frames(paths: Sequence[Path]) -> list[pd.DataFrame]:
    return [load_table(path, dtype=str).fillna("") for path in paths]


def _extract_column(frame: pd.DataFrame, column_lower: str) -> str:
    mapping = {col.lower(): col for col in frame.columns}
    try:
        return mapping[column_lower]
    except KeyError as exc:  # pragma: no cover - validated upstream
        raise AdjudicationError(f"Column '{column_lower}' not present") from exc


def _prepare_metadata(
    frames: Sequence[pd.DataFrame], id_lower: str, decision_lower: str
) -> tuple[dict[str, dict[str, str]], list[str]]:
    if not frames:
        return {}, []

    first_frame = frames[0]
    id_column = _extract_column(first_frame, id_lower)
    metadata_columns = [
        column for column in first_frame.columns if column.lower() != decision_lower
    ]

    metadata: dict[str, dict[str, str]] = {}
    for row in first_frame.itertuples(index=False):
        item_id = str(getattr(row, id_column))
        if not item_id:
            continue
        metadata[item_id] = {
            column: str(getattr(row, column, "")) for column in metadata_columns
        }
        metadata[item_id]["item_id"] = item_id
    return metadata, metadata_columns


def _aggregate_decisions(
    frames: Sequence[pd.DataFrame],
    rater_names: Sequence[str],
    id_lower: str,
    decision_lower: str,
    metadata: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    decision_map: dict[str, dict[str, str]] = {}

    for frame, rater in zip(frames, rater_names):
        id_column = _extract_column(frame, id_lower)
        decision_column = _extract_column(frame, decision_lower)
        for row in frame.itertuples(index=False):
            item_id = str(getattr(row, id_column))
            if not item_id:
                continue
            try:
                normalized = _normalize_decision(getattr(row, decision_column))
            except AdjudicationError:
                raise
            decision_map.setdefault(item_id, {})[rater] = normalized
            if item_id not in metadata:
                metadata[item_id] = {
                    column: str(getattr(row, column, ""))
                    for column in frame.columns
                    if column.lower() != decision_lower
                }
                metadata[item_id]["item_id"] = item_id
    return decision_map


def build_adjudication_items(
    csv_paths: Sequence[Path],
    rater_names: Sequence[str],
    id_column: str | None,
    decision_column: str | None,
) -> tuple[list[AdjudicationItem], list[str]]:
    frames = _load_frames(csv_paths)
    if len(frames) < 2:
        raise AdjudicationError("At least two CSV files are required for adjudication.")

    id_lower = _infer_common_column(
        frames, id_column, _ID_CANDIDATES, description="Identifier"
    )
    decision_lower = _infer_common_column(
        frames, decision_column, _DECISION_CANDIDATES, description="Decision"
    )

    metadata, metadata_columns = _prepare_metadata(frames, id_lower, decision_lower)
    decisions = _aggregate_decisions(
        frames, rater_names, id_lower, decision_lower, metadata
    )

    items: list[AdjudicationItem] = []
    for item_id, meta in metadata.items():
        rater_votes = decisions.get(item_id, {})
        decision_values = set(rater_votes.values())
        needs_review = (
            len(decision_values) > 1 or len(rater_votes) < len(rater_names)
        )
        if needs_review:
            items.append(
                AdjudicationItem(
                    item_id=item_id,
                    metadata=meta,
                    rater_decisions=rater_votes,
                )
            )

    items.sort(key=lambda item: item.item_id)
    return items, metadata_columns


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def wrap_text(text: str, width: int = 70) -> Iterator[str]:
    import textwrap

    if text is None:
        return iter(())

    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            paragraph, width=width, replace_whitespace=False, drop_whitespace=False
        )
        if not wrapped:
            lines.append("")
        else:
            lines.extend(wrapped)
    return iter(lines)


def present_item(
    item: AdjudicationItem,
    index: int,
    total: int,
    rater_names: Sequence[str],
) -> None:
    clear_screen()
    header_text = f"🗳️  Adjudicating Item {index + 1} of {total}"
    minimum_width = max(len(header_text) + 4, 70)
    terminal_width = shutil.get_terminal_size(fallback=(minimum_width, 24)).columns
    content_width = max(minimum_width, terminal_width)
    divider = "─" * content_width

    print(f"╭{'─' * (content_width - 2)}╮")
    print(f"│{header_text.center(content_width - 2)}│")
    print(f"╰{'─' * (content_width - 2)}╯")
    print()

    indent = "  "
    label_width = 16
    value_width = max(20, content_width - len(indent) - label_width - 3)

    def print_field(label: str, value: str) -> None:
        clean_value = value or ""
        lines = list(wrap_text(clean_value, width=value_width)) or [""]
        label_text = f"{label:<{label_width}}"
        first, *rest = lines
        print(f"{indent}{label_text} : {first}")
        for line in rest:
            print(f"{indent}{' ' * label_width}   {line}")

    lower_map = {col.lower(): col for col in item.metadata}

    def emit_if_present(lower_name: str, display_label: str | None = None) -> None:
        column = lower_map.get(lower_name)
        if column:
            label_text = display_label or column
            print_field(label_text, item.metadata.get(column, ""))

    print("📁 Item Details")
    print(divider)
    emit_if_present("item_id", "Item ID")
    emit_if_present("row_index")
    emit_if_present("dataset")
    emit_if_present("label")
    emit_if_present("name")
    emit_if_present("description")
    print()

    if any(key in lower_map for key in ["gold_labels", "gold_definitions", "gold_paths"]):
        print("🏷️  Ground Truth")
        print(divider)
        emit_if_present("gold_labels", "Labels")
        emit_if_present("gold_paths", "Paths")
        emit_if_present("gold_definitions", "Definitions")
        print()

    if any(key in lower_map for key in ["resolved_keywords", "resolved_paths", "resolved_definitions"]):
        print("🤖 Model Prediction")
        print(divider)
        emit_if_present("resolved_keywords", "Resolved Keywords")
        emit_if_present("resolved_paths", "Paths")
        emit_if_present("resolved_definitions", "Definitions")
        print()

    print("🧮 Rater Decisions")
    print(divider)
    for rater in rater_names:
        decision = item.rater_decisions.get(rater, "(missing)")
        print_field(rater, decision)
    print(divider)
    print("Options: [A] Accept   [X] Reject   [U] Unsure   [B] Back   [Q] Quit")
    print(divider)


def append_decision(
    output_path: Path,
    row: dict[str, str],
    field_order: Sequence[str],
) -> None:
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(field_order))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_decisions(output_path: Path) -> dict[str, str]:
    if not output_path.exists():
        return {}
    frame = load_table(output_path, dtype=str).fillna("")
    decisions: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        item_id = str(getattr(row, "item_id", ""))
        final = str(getattr(row, "final_decision", ""))
        if item_id:
            decisions[item_id] = final
    return decisions


def review_items(
    items: Sequence[AdjudicationItem],
    existing: dict[str, str],
    metadata_columns: Sequence[str],
    rater_names: Sequence[str],
    output_path: Path,
) -> None:
    total = len(items)
    if total == 0:
        print("All items already have unanimous decisions. Nothing to adjudicate.")
        return

    field_order = [
        "item_id",
        *metadata_columns,
        *[f"{name}_decision" for name in rater_names],
        "final_decision",
    ]

    index = 0
    while index < total:
        item = items[index]
        if item.item_id in existing:
            index += 1
            continue

        present_item(item, index, total, rater_names)
        try:
            choice = input("Final decision: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSession terminated. Progress saved to", output_path)
            return

        if not choice:
            continue
        if choice in ADJUDICATION_KEYS:
            decision = ADJUDICATION_KEYS[choice]
            row = item.to_output_row(decision, metadata_columns, rater_names)
            append_decision(output_path, row, field_order)
            existing[item.item_id] = decision
            index += 1
        elif choice == "b":
            index = max(index - 1, 0)
        elif choice == "q":
            print("Exiting. Progress saved in", output_path)
            return
        else:
            print("Unrecognized option. Please choose A, X, U, B, or Q.")

    print("Adjudication complete. Decisions saved to", output_path)


def _resolve_rater_names(paths: Sequence[Path], requested: Iterable[str] | None) -> list[str]:
    if requested:
        names = list(requested)
        if len(names) != len(paths):
            raise AdjudicationError(
                "Number of rater names must match number of CSV inputs."
            )
        return names
    return [path.stem for path in paths]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_paths = [path.resolve() for path in args.csv_paths]
    rater_names = _resolve_rater_names(csv_paths, args.rater_names)

    try:
        items, metadata_columns = build_adjudication_items(
            csv_paths, rater_names, args.id_column, args.decision_column
        )
    except AdjudicationError as exc:
        print(f"Error: {exc}")
        return

    existing_decisions = load_existing_decisions(args.output)
    review_items(items, existing_decisions, metadata_columns, rater_names, args.output)


if __name__ == "__main__":
    main()
