from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List


def _find_column(columns: Iterable[str], suffix: str) -> str | None:
    suffix = suffix.lower()
    for column in columns:
        if column.lower().endswith(suffix):
            return column
    return None


def _parse_keywords(raw: str | None) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = text
        else:
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
    if text.startswith('"""') and text.endswith('"""'):
        inner = text[3:-3].replace('""', '"')
        return [inner] if inner.strip() else []
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].replace('""', '"')
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        return [part for part in parts if part]
    return [text]


def build_report(
    rows: list[dict[str, str]],
    *,
    correct_col: str,
    gold_col: str,
    pred_col: str,
    top_gold: int,
    top_preds: int,
) -> str:
    mistakes = [
        row
        for row in rows
        if row.get(correct_col, "").strip().lower() in ("false", "0", "no")
    ]

    gold_counter: Counter[str] = Counter()
    pred_counter: Counter[str] = Counter()
    per_gold_pred: dict[str, Counter[str]] = defaultdict(Counter)

    for row in mistakes:
        golds = _parse_keywords(row.get(gold_col))
        preds = _parse_keywords(row.get(pred_col))
        gold_counter.update(golds)
        pred_counter.update(preds)
        for gold in golds:
            for pred in preds:
                per_gold_pred[gold][pred] += 1

    lines: list[str] = []
    lines.append("# Confusion report")
    lines.append("")
    lines.append(f"Total rows: {len(rows):,}")
    lines.append(f"Mistakes: {len(mistakes):,}")
    lines.append("")

    lines.append("## Top mistaken gold labels")
    lines.append("")
    lines.append("| Gold label | Mistakes |")
    lines.append("| --- | ---: |")
    for gold, count in gold_counter.most_common(top_gold):
        lines.append(f"| {gold} | {count} |")
    lines.append("")

    lines.append("## Top predicted labels (wrong)")
    lines.append("")
    lines.append("| Predicted label | Count |")
    lines.append("| --- | ---: |")
    for pred, count in pred_counter.most_common(top_preds):
        lines.append(f"| {pred} | {count} |")
    lines.append("")

    lines.append("## Confusions by gold label")
    lines.append("")
    for gold, _ in gold_counter.most_common(top_gold):
        lines.append(f"### {gold}")
        lines.append("")
        lines.append("| Predicted label | Count |")
        lines.append("| --- | ---: |")
        for pred, count in per_gold_pred[gold].most_common(top_preds):
            lines.append(f"| {pred} | {count} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a confusion report from a results CSV."
    )
    parser.add_argument("input_csv", type=Path, help="Path to results CSV.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Destination markdown file.",
    )
    parser.add_argument("--top-gold", type=int, default=15)
    parser.add_argument("--top-preds", type=int, default=10)
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows or not reader.fieldnames:
            raise ValueError("CSV appears to be empty.")
        columns = reader.fieldnames

    correct_col = _find_column(columns, "correct")
    gold_col = _find_column(columns, "gold_labels")
    pred_col = _find_column(columns, "resolved_keywords")
    missing = [name for name, col in [
        ("correct", correct_col),
        ("gold_labels", gold_col),
        ("resolved_keywords", pred_col),
    ] if col is None]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")

    report = build_report(
        rows,
        correct_col=correct_col,
        gold_col=gold_col,
        pred_col=pred_col,
        top_gold=args.top_gold,
        top_preds=args.top_preds,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
