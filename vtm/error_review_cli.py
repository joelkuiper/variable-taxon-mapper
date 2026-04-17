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
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import pandas as pd

from vtm.config import TaxonomyFieldMappingConfig, load_config
from vtm.pipeline.service import prepare_keywords_dataframe
from vtm.utils import load_table, resolve_prefixed_column
from .taxonomy import build_name_maps_from_graph, build_taxonomy_graph


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
    gold_paths: List[str]
    resolved_keywords: List[str]
    resolved_definitions: List[str]
    resolved_paths: List[str]

    def to_output_row(self, decision: str) -> dict[str, str]:
        """Return a dictionary suitable for CSV persistence."""

        return {
            "row_index": str(self.row_index),
            "dataset": self.dataset,
            "label": self.label,
            "name": self.name,
            "description": self.description,
            "gold_labels": " | ".join(self.gold_labels),
            "gold_definitions": "\n".join(self.gold_definitions),
            "resolved_keywords": " | ".join(self.resolved_keywords),
            "resolved_definitions": " | ".join(self.resolved_definitions),
            "resolved_paths": " | ".join(self.resolved_paths),
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
        "--config",
        type=Path,
        help=(
            "Optional path to a TOML config file; column mappings will be "
            "read from its [taxonomy_fields] section."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("error_review_decisions.csv"),
        help="Destination CSV file for storing review decisions.",
    )
    return parser.parse_args(argv)


def load_keywords_metadata(
    keywords_path: Path,
    taxonomy_fields: TaxonomyFieldMappingConfig,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return keyword definitions and taxonomy paths keyed by keyword name."""

    raw_df = load_table(keywords_path, low_memory=False)
    canonical_df, definition_df, multi_parents = prepare_keywords_dataframe(
        raw_df, taxonomy_fields
    )

    # In case of duplicate names, keep the first non-empty definition.
    definitions: dict[str, str] = {}
    if definition_df is not None:
        for row in definition_df.fillna("").itertuples(index=False):
            name = str(row.name).strip()
            summary = str(row.definition).strip()
            if not name:
                continue
            if name not in definitions or not definitions[name]:
                definitions[name] = summary
    elif "definition" in canonical_df.columns:
        for _, row in canonical_df.iterrows():
            name = str(row.get("name", "")).strip()
            definition = str(row.get("definition", "")).strip()
            if not name:
                continue
            if name not in definitions or not definitions[name]:
                definitions[name] = definition

    taxonomy_paths: dict[str, str] = {}
    if {"name", "parent"}.issubset(canonical_df.columns):
        taxonomy_df = canonical_df.copy()
        if "parent" in taxonomy_df.columns:
            taxonomy_df["parent"] = taxonomy_df["parent"].apply(
                lambda value: value if pd.notna(value) and str(value).strip() else pd.NA
            )
        try:
            graph = build_taxonomy_graph(
                taxonomy_df, multi_parents=multi_parents
            )
        except Exception:
            graph = None
        if graph is not None:
            _, taxonomy_paths = build_name_maps_from_graph(graph)

    return definitions, taxonomy_paths


def load_prediction_errors(
    predictions_path: Path, *, resolved_column: str | None = None
) -> pd.DataFrame:
    """Return a DataFrame containing only misclassified rows."""

    df = load_table(predictions_path, dtype=str).fillna("")
    if resolved_column:
        resolved_col = resolve_prefixed_column(df.columns, resolved_column)
        if resolved_col and resolved_col != resolved_column:
            df = df.rename(columns={resolved_col: resolved_column})
        paths_col = resolve_prefixed_column(df.columns, "resolved_paths")
        if paths_col and paths_col != "resolved_paths":
            df = df.rename(columns={paths_col: "resolved_paths"})
        defs_col = resolve_prefixed_column(df.columns, "resolved_definitions")
        if defs_col and defs_col != "resolved_definitions":
            df = df.rename(columns={defs_col: "resolved_definitions"})
    correct_col = resolve_prefixed_column(df.columns, "correct")
    if correct_col is None:
        raise KeyError("Predictions file is missing the 'correct' column.")
    # Normalize the `correct` column into booleans.
    normalized = df[correct_col].astype(str).str.lower()
    mask = normalized.isin({"false", "0", "no"})
    errors = df[mask].copy()
    if correct_col != "correct":
        errors = errors.rename(columns={correct_col: "correct"})
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
    errors: pd.DataFrame,
    definitions: dict[str, str],
    taxonomy_paths: dict[str, str],
) -> List[ErrorRecord]:
    records: List[ErrorRecord] = []
    resolved_col = resolve_prefixed_column(errors.columns, "resolved_keywords")
    paths_col = resolve_prefixed_column(errors.columns, "resolved_paths")
    defs_col = resolve_prefixed_column(errors.columns, "resolved_definitions")
    for row in errors.itertuples():
        gold_labels = parse_label_list(getattr(row, "gold_labels", ""))
        gold_definitions = [definitions.get(label, "") for label in gold_labels]
        gold_paths = [taxonomy_paths.get(label, "") for label in gold_labels]
        resolved_keywords = (
            parse_label_list(str(getattr(row, resolved_col, ""))) if resolved_col else []
        )
        resolved_definitions = (
            parse_label_list(str(getattr(row, defs_col, ""))) if defs_col else []
        )
        if not resolved_definitions and resolved_keywords:
            resolved_definitions = [definitions.get(label, "") for label in resolved_keywords]
        resolved_paths = (
            parse_label_list(str(getattr(row, paths_col, ""))) if paths_col else []
        )

        records.append(
            ErrorRecord(
                row_index=int(getattr(row, "row_index")),
                dataset=str(getattr(row, "dataset", "")),
                label=str(getattr(row, "label", "")),
                name=str(getattr(row, "name", "")),
                description=str(getattr(row, "description", "")),
                gold_labels=gold_labels,
                gold_definitions=gold_definitions,
                gold_paths=gold_paths,
                resolved_keywords=resolved_keywords,
                resolved_definitions=resolved_definitions,
                resolved_paths=resolved_paths,
            )
        )
    return records


def load_existing_decisions(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    df = load_table(path, dtype=str).fillna("")
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
    header_text = f"🔎  Reviewing Error {index + 1} of {total}"
    minimum_width = max(len(header_text) + 4, 62)
    terminal_width = shutil.get_terminal_size(fallback=(minimum_width, 24)).columns
    content_width = max(minimum_width, terminal_width)
    divider = "─" * content_width

    print(f"╭{'─' * (content_width - 2)}╮")
    print(f"│{header_text.center(content_width - 2)}│")
    print(f"╰{'─' * (content_width - 2)}╯")
    print()

    label_width = 13
    indent = "  "
    value_width = max(20, content_width - len(indent) - label_width - 3)

    def print_field(label: str, value: str) -> None:
        clean_value = value or ""
        lines = list(wrap_text(clean_value, width=value_width)) or [""]
        label_text = f"{label:<{label_width}}"
        first, *rest = lines
        print(f"{indent}{label_text} : {first}")
        for line in rest:
            print(f"{indent}{' ' * label_width}   {line}")

    print("📁 Item")
    print(divider)
    print_field("Dataset", record.dataset)
    print_field("Label", record.label)
    print_field("Name", record.name)
    print_field("Description", record.description)
    print()

    print("🏷️  Ground Truth")
    print(divider)
    if record.gold_labels:
        for label, path, definition in zip(
            record.gold_labels, record.gold_paths, record.gold_definitions
        ):
            print_field("Label", label)
            print_field("Path", path or "(no path)")
            print_field("Definition", definition or "(no definition)")
            print()
    else:
        print(f"{indent}(no gold labels provided)")
        print()

    print("🤖 Model Prediction")
    print(divider)
    resolved_text = " | ".join(record.resolved_keywords)
    print_field("Resolved Keywords", resolved_text or "(none)")
    paths_text = " | ".join(record.resolved_paths)
    print_field("Paths", paths_text or "(none)")
    if record.resolved_definitions:
        print_field("Definitions", " | ".join(record.resolved_definitions))
    else:
        print_field("Definition", "(no definition)")
    print(divider)
    print("Options: [A] Accept   [X] Reject   [U] Unknown   [B] Back   [Q] Quit")
    print(divider)


def wrap_text(text: str, width: int = 70) -> Iterator[str]:
    import textwrap

    if text is None:
        return iter(())

    lines: List[str] = []
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

    taxonomy_fields = TaxonomyFieldMappingConfig()
    if args.config is not None:
        config_path = args.config.resolve()
        config = load_config(config_path)
        taxonomy_fields = config.taxonomy_fields

    keywords_path = args.keywords
    if args.config is not None and not keywords_path.is_absolute():
        keywords_path = (args.config.parent / keywords_path).resolve()
    else:
        keywords_path = keywords_path.resolve()

    definitions, taxonomy_paths = load_keywords_metadata(
        keywords_path, taxonomy_fields
    )
    errors = load_prediction_errors(args.predictions, resolved_column="resolved_keywords")
    records = enrich_records(errors, definitions, taxonomy_paths)
    existing_decisions = load_existing_decisions(args.output)

    review_records(records, existing_decisions, args.output)


if __name__ == "__main__":
    main()
