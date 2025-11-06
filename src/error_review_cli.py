"""Command-line interface for reviewing model prediction errors.

This module loads model prediction outputs alongside keyword definitions and
provides an interactive terminal experience for quickly triaging misclassified
items.  Users can mark each error as an acceptable mistake, a clear rejection,
or unknown/unclear.  Decisions are persisted to disk so that review sessions can
be resumed at any time.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import pandas as pd


DECISION_KEYS = {
    "a": "accept",
    "x": "reject",
    "u": "unknown",
}

@dataclass
class ErrorRecord:
    """Structured information about a single model error."""

    row_index: int
    dataset: str
    label: str
    name: str
    description: str
    gold_labels: List[str]
    gold_definitions: List[str]
    resolved_label: str
    resolved_definition: str
    resolved_path: str

    def to_output_row(self, decision: str) -> dict[str, str]:
        """Return a dictionary suitable for CSV persistence."""

        return {
            "row_index": str(self.row_index),
            "dataset": self.dataset,
            "label": self.label,
            "name": self.name,
            "description": self.description,
            "gold_labels": " | ".join(self.gold_labels),
            "gold_definition_summaries": "\n".join(self.gold_definitions),
            "resolved_label": self.resolved_label,
            "resolved_definition_summary": self.resolved_definition,
            "resolved_path": self.resolved_path,
            "decision": decision,
        }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive CLI for reviewing model prediction errors. "
            "Decisions are saved to an output CSV for later analysis."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to a CSV file containing model predictions.",
    )
    parser.add_argument(
        "--keywords",
        required=True,
        type=Path,
        help="Path to the Keywords.csv file providing definitions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("error_review_decisions.csv"),
        help="Destination CSV file for storing review decisions.",
    )
    return parser.parse_args(argv)


def load_keywords_definitions(keywords_path: Path) -> dict[str, str]:
    """Load keyword definitions keyed by the `name` column."""

    df = pd.read_csv(keywords_path, dtype=str).fillna("")
    # In case of duplicate names, keep the first non-empty definition summary.
    definitions: dict[str, str] = {}
    for row in df.itertuples():
        name = str(row.name)
        definition = str(getattr(row, "definition_summary", ""))
        if name not in definitions or not definitions[name]:
            definitions[name] = definition
    return definitions


def load_prediction_errors(predictions_path: Path) -> pd.DataFrame:
    """Return a DataFrame containing only misclassified rows."""

    df = pd.read_csv(predictions_path, dtype=str).fillna("")
    # Normalize the `correct` column into booleans.
    normalized = df["correct"].astype(str).str.lower()
    mask = normalized.isin({"false", "0", "no"})
    errors = df[mask].copy()
    errors.reset_index(inplace=True)
    errors.rename(columns={"index": "row_index"}, inplace=True)
    return errors


def parse_label_list(value: str) -> List[str]:
    """Parse a representation of label collections into a list of strings."""

    if not value:
        return []
    value = value.strip()
    if not value:
        return []
    # Attempt to parse JSON-like list representations using ast.literal_eval.
    if value.startswith("[") and value.endswith("]"):
        import ast

        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
        else:
            return [str(item).strip() for item in parsed if str(item).strip()]
    # Fall back to splitting on common delimiters.
    for delimiter in ["|", ";", ","]:
        if delimiter in value:
            parts = [part.strip() for part in value.split(delimiter)]
            return [part for part in parts if part]
    return [value]


def enrich_records(
    errors: pd.DataFrame, definitions: dict[str, str]
) -> List[ErrorRecord]:
    records: List[ErrorRecord] = []
    for row in errors.itertuples():
        gold_labels = parse_label_list(getattr(row, "gold_labels", ""))
        gold_definitions = [definitions.get(label, "") for label in gold_labels]
        resolved_label = str(getattr(row, "resolved_label", ""))
        resolved_definition = definitions.get(resolved_label, "")

        records.append(
            ErrorRecord(
                row_index=int(getattr(row, "row_index")),
                dataset=str(getattr(row, "dataset", "")),
                label=str(getattr(row, "label", "")),
                name=str(getattr(row, "name", "")),
                description=str(getattr(row, "description", "")),
                gold_labels=gold_labels,
                gold_definitions=gold_definitions,
                resolved_label=resolved_label,
                resolved_definition=resolved_definition,
                resolved_path=str(getattr(row, "resolved_path", "")),
            )
        )
    return records


def load_existing_decisions(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    decisions: dict[int, str] = {}
    for row in df.itertuples():
        try:
            row_index = int(getattr(row, "row_index"))
        except (TypeError, ValueError):
            continue
        decisions[row_index] = str(getattr(row, "decision", ""))
    return decisions


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def present_record(record: ErrorRecord, index: int, total: int) -> None:
    clear_screen()
    header = f"Reviewing error {index + 1} of {total}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print()

    context_lines = [
        f"Dataset       : {record.dataset}",
        f"Label         : {record.label}",
        f"Name          : {record.name}",
        f"Description   : {record.description}",
    ]
    print("Context:")
    for line in context_lines:
        print(f"  {line}")
    print()

    gold_lines = []
    for label, definition in zip(record.gold_labels, record.gold_definitions):
        gold_lines.append(f"{label}")
        if definition:
            for wrapped in wrap_text(definition):
                gold_lines.append(f"    {wrapped}")
        else:
            gold_lines.append("    (no definition)")
        gold_lines.append("")
    if gold_lines:
        print("Ground Truth:")
        for line in gold_lines:
            print(f"  {line}")
    else:
        print("Ground Truth:\n  (no gold labels provided)")
    print()

    print("Model Prediction:")
    print(f"  Resolved Label : {record.resolved_label or '(none)'}")
    if record.resolved_definition:
        print("  Definition     :")
        for wrapped in wrap_text(record.resolved_definition):
            print(f"    {wrapped}")
    else:
        print("  Definition     : (no definition)")
    print(f"  Resolved Path  : {record.resolved_path or '(none)'}")
    print()

    print("Options: [A]ccept  [X] Reject  [U]nknown  [B]ack  [Q]uit")


def wrap_text(text: str, width: int = 70) -> Iterator[str]:
    import textwrap

    return iter(textwrap.wrap(text, width=width, replace_whitespace=False))


def append_decision(output_path: Path, row: dict[str, str]) -> None:
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def review_records(
    records: List[ErrorRecord],
    existing_decisions: dict[int, str],
    output_path: Path,
) -> None:
    total = len(records)
    if total == 0:
        print("No errors found in the provided predictions file.")
        return

    index = 0
    while index < total:
        record = records[index]
        if record.row_index in existing_decisions:
            index += 1
            continue

        present_record(record, index, total)

        try:
            choice = input("Decision: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSession terminated by user. Progress saved.")
            return

        if not choice:
            continue
        if choice in DECISION_KEYS:
            decision = DECISION_KEYS[choice]
            row = record.to_output_row(decision)
            append_decision(output_path, row)
            existing_decisions[record.row_index] = decision
            index += 1
        elif choice == "b":
            index = max(index - 1, 0)
        elif choice == "q":
            print("Exiting. Progress saved in", output_path)
            return
        else:
            print("Unrecognized option. Please choose A, X, U, B, or Q.")

    print("All errors have been reviewed. Decisions saved to", output_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)

    definitions = load_keywords_definitions(args.keywords)
    errors = load_prediction_errors(args.predictions)
    records = enrich_records(errors, definitions)
    existing_decisions = load_existing_decisions(args.output)

    review_records(records, existing_decisions, args.output)


if __name__ == "__main__":
    main()
