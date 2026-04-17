from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd


def _humanize(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").title()


def _display_match_type(label: str) -> str:
    normalized = str(label).strip()
    if normalized.lower() == "none":
        return "Wrong"
    if normalized == "(missing)":
        return "Missing Match Type"
    return _humanize(normalized)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))  # type: ignore[arg-type]
    except TypeError:
        return False


def _as_int(value: Any) -> str:
    if _is_missing(value):
        return "–"
    return f"{int(value):,}"


def _as_float(value: Any, digits: int = 4) -> str:
    if _is_missing(value):
        return "–"
    return f"{float(value):.{digits}f}"


def _as_percent(value: Any) -> str:
    if _is_missing(value):
        return "–"
    return f"{float(value) * 100:.2f}%"


def _markdown(df_obj: pd.DataFrame, *, index: bool = False) -> str:
    if df_obj.empty:
        return ""
    return df_obj.to_markdown(index=index, tablefmt="rounded_grid")


def _dataframe_to_text(frame: pd.DataFrame) -> str:
    try:
        return frame.to_string(index=False)
    except Exception:  # pragma: no cover - defensive fallback
        return frame.to_string()


def format_metrics(
    metrics: dict[str, Any] | None,
    df: pd.DataFrame | None = None,
    *,
    dataset_column: str | None = "dataset",
) -> str:
    sections: list[str] = []

    sections.extend(["## Evaluation Metrics", ""])
    consumed: set[str] = set()

    metrics_map: Mapping[str, Any] = metrics or {}

    summary_table = _build_summary_section(metrics_map, consumed)
    if summary_table:
        sections.extend(["### Summary", summary_table, ""])

    accuracy_table = _build_accuracy_section(metrics_map, consumed)
    if accuracy_table:
        sections.extend(["### Accuracy", accuracy_table, ""])

    match_table = _build_match_type_section(metrics_map, consumed)
    if match_table:
        sections.extend(["### Match types", match_table, ""])

    strategy_table = _build_strategy_section(metrics_map, consumed)
    if strategy_table:
        sections.extend(["### Match strategy performance", strategy_table, ""])

    hierarchy_table = _build_hierarchical_section(metrics_map, consumed)
    if hierarchy_table:
        sections.extend(["### Hierarchical distance metrics", hierarchy_table, ""])

    # dataset_table = _build_dataset_section(df, dataset_column=dataset_column)
    # if dataset_table:
    #     sections.extend(["### Performance by dataset", dataset_table, ""])

    # additional_table = _build_additional_metrics_section(metrics, consumed)
    # if additional_table:
    #     sections.extend(["### Additional metrics", additional_table, ""])

    top_sections = _build_top_error_sections(df)
    sections.extend(top_sections)

    return "\n".join(part for part in sections if part).strip()


def _build_summary_section(metrics: Mapping[str, Any], consumed: set[str]) -> str:
    rows: list[dict[str, str]] = []

    def add(label: str, key: str, kind: str) -> None:
        if key not in metrics:
            return
        value = metrics[key]
        formatters: dict[str, Callable[[Any], str]] = {
            "int": _as_int,
            "pct": _as_percent,
            "float": _as_float,
        }
        formatter = formatters.get(kind)
        formatted = formatter(value) if formatter is not None else str(value)
        rows.append({"Metric": label, "Value": formatted})
        consumed.add(key)

    add("Rows after dedupe", "n_total_rows_after_dedupe", "int")
    add("Rows with gold-label values", "n_with_any_gold_label", "int")
    add("Eligible rows", "n_eligible", "int")
    add("Excluded (not in taxonomy)", "n_excluded_not_in_taxonomy", "int")
    add("Evaluated rows", "n_evaluated", "int")
    # add("Errors", "n_errors", "int")
    add("Possible correct under allowed", "n_possible_correct_under_allowed", "int")
    add(
        "Possible correct under allowed rate",
        "possible_correct_under_allowed_rate",
        "pct",
    )
    add("Predictions correct", "n_correct", "int")
    n_evaluated = metrics.get("n_evaluated")
    n_correct = metrics.get("n_correct")
    if n_evaluated is not None and n_correct is not None:
        try:
            wrong = int(n_evaluated) - int(n_correct)
        except (TypeError, ValueError):
            wrong = None
        if wrong is not None:
            rows.append({"Metric": "Predictions wrong", "Value": _as_int(wrong)})

    if not rows:
        return ""

    summary_df = pd.DataFrame(rows)
    return _markdown(summary_df)


def _build_accuracy_section(metrics: Mapping[str, Any], consumed: set[str]) -> str:
    rows: list[dict[str, str]] = []

    def add(label: str, key: str) -> None:
        if key not in metrics:
            return
        value = metrics[key]
        rows.append({"Metric": label, "Value": _as_percent(value)})
        consumed.add(key)

    add("Accuracy (any match)", "label_accuracy_any_match")
    add("Accuracy (exact only)", "label_accuracy_exact_only")
    add("Accuracy (ancestor only)", "label_accuracy_ancestor_only")
    add("Accuracy (descendant only)", "label_accuracy_descendant_only")

    if not rows:
        return ""

    accuracy_df = pd.DataFrame(rows)
    return _markdown(accuracy_df)


def _build_match_type_section(metrics: Mapping[str, Any], consumed: set[str]) -> str:
    counts = metrics.get("match_type_counts") or {}
    rates = metrics.get("match_type_rates") or {}

    if not counts and not rates:
        return ""

    rows: list[dict[str, str]] = []
    keys = sorted({*counts.keys(), *rates.keys()})
    for key in keys:
        display_key = "wrong" if key == "none" else key
        rows.append(
            {
                "Match type": _humanize(str(display_key)),
                "Count": _as_int(counts.get(key)),
                "Rate": _as_percent(rates.get(key)),
            }
        )

    consumed.update({"match_type_counts", "match_type_rates"})

    match_df = pd.DataFrame(rows)
    return _markdown(match_df)


def _build_hierarchical_section(metrics: Mapping[str, Any], consumed: set[str]) -> str:
    # Human-readable labels for hierarchical metrics
    _HUMAN_LABELS: dict[str, tuple[str, str]] = {
        "hierarchical_distance_count": ("Rows with computable distances", "int"),
        "hierarchical_distance_error_count": (
            "Errors with computable distances",
            "int",
        ),
        "hierarchical_distance_error_mean": ("Mean error distance (edges)", "float"),
        "hierarchical_distance_error_median": (
            "Median error distance (edges)",
            "float",
        ),
        "hierarchical_distance_error_within_1_rate": ("Error distance ≤ 1 step", "pct"),
        "hierarchical_distance_error_within_2_rate": (
            "Error distance ≤ 2 steps",
            "pct",
        ),
        "hierarchical_distance_min_mean": (
            "Mean minimal distance to any gold",
            "float",
        ),
        "hierarchical_distance_min_median": (
            "Median minimal distance to any gold",
            "float",
        ),
        "hierarchical_distance_within_1_rate": ("Minimal distance ≤ 1 step", "pct"),
        "hierarchical_distance_within_2_rate": ("Minimal distance ≤ 2 steps", "pct"),
    }

    rows: list[dict[str, str]] = []
    for key, (human_label, kind) in _HUMAN_LABELS.items():
        if key not in metrics:
            continue
        value = metrics[key]
        formatter = {"int": _as_int, "pct": _as_percent, "float": _as_float}[kind]
        rows.append({"Metric": human_label, "Value": formatter(value)})
        consumed.add(key)

    if not rows:
        return ""

    hier_df = pd.DataFrame(rows)
    return _markdown(hier_df)


def _build_strategy_section(metrics: Mapping[str, Any], consumed: set[str]) -> str:
    strategy_perf: dict[str, Any] | None = metrics.get("match_strategy_performance")
    strategy_share: dict[str, Any] | None = metrics.get("match_strategy_correct_share")
    strategy_volume: dict[str, Any] | None = metrics.get("match_strategy_volume")

    if not isinstance(strategy_perf, dict) or not strategy_perf:
        other_metrics = metrics.get("other_metrics")
        if isinstance(other_metrics, dict):
            fallback_perf = other_metrics.get("match_strategy_performance")
            if isinstance(fallback_perf, dict) and fallback_perf:
                strategy_perf = fallback_perf
                strategy_share = other_metrics.get("match_strategy_correct_share")
                strategy_volume = other_metrics.get("match_strategy_volume")

    if not isinstance(strategy_perf, dict) or not strategy_perf:
        return ""

    if not isinstance(strategy_share, dict):
        strategy_share = {}
    if not isinstance(strategy_volume, dict):
        strategy_volume = {}

    rows: list[dict[str, str]] = []
    for key in sorted(strategy_perf):
        perf = strategy_perf.get(key) or {}
        rows.append(
            {
                "Strategy": _humanize(str(key)),
                "Volume": _as_int(perf.get("n", strategy_volume.get(key))),
                "Accuracy": _as_percent(perf.get("accuracy")),
                "Correct": _as_int(perf.get("n_correct")),
                "Correct share": _as_percent(strategy_share.get(key)),
            }
        )

    consumed.update(
        {
            "match_strategy_performance",
            "match_strategy_correct_share",
            "match_strategy_volume",
        }
    )

    strategy_df = pd.DataFrame(rows)
    return _markdown(strategy_df)


def _build_additional_metrics_section(
    metrics: dict[str, Any], consumed: set[str]
) -> str:
    remaining = {key: value for key, value in metrics.items() if key not in consumed}
    other_raw = remaining.pop("other_metrics", None)
    if isinstance(other_raw, dict):
        for key, value in other_raw.items():
            if key.startswith("hierarchical_distance_"):
                continue
            if key.startswith("match_strategy_"):
                continue
            remaining[key] = value

    if not remaining:
        return ""

    rows: list[dict[str, str]] = []
    for key, value in sorted(remaining.items()):
        if isinstance(value, dict):
            formatted = json.dumps(value, indent=2, sort_keys=True)
            rows.append({"Metric": _humanize(str(key)), "Value": f"`{formatted}`"})
            continue

        key_str = str(key)
        if key_str.endswith("_rate"):
            formatted = _as_percent(value)
        elif key_str.startswith("n_"):
            formatted = _as_int(value)
        else:
            formatted = _as_float(value)
        rows.append({"Metric": _humanize(str(key)), "Value": formatted})

    additional_df = pd.DataFrame(rows)
    return _markdown(additional_df)


def _build_dataset_section(
    df: pd.DataFrame | None, *, dataset_column: str | None
) -> str:
    if (
        df is None
        or dataset_column is None
        or dataset_column not in df.columns
        or "correct" not in df.columns
    ):
        return ""

    grouped = df.groupby(dataset_column, dropna=False)
    rows: list[dict[str, Any]] = []
    has_possible = "possible_correct_under_allowed" in df.columns
    has_match_type = "match_type" in df.columns

    match_type_values: Sequence[str] = []
    if has_match_type:
        normalized_types = (
            df["match_type"].fillna("(missing)").astype(str)
            if not df["match_type"].empty
            else pd.Series(dtype=str)
        )
        match_type_values = sorted(normalized_types.unique())

    for dataset, group in grouped:
        evaluated = len(group)
        if evaluated == 0:
            continue

        correct_series = group["correct"].fillna(False).astype(bool)
        if correct_series.empty:
            continue

        correct_count = int(correct_series.sum())
        accuracy = float(correct_series.mean())

        row: dict[str, Any] = {
            "Dataset": str(dataset) if dataset is not None else "(missing)",
            "Evaluated": evaluated,
            "Correct": correct_count,
            "Accuracy": accuracy,
        }

        if has_possible:
            possible_series = group["possible_correct_under_allowed"].fillna(False)
            if not possible_series.empty:
                possible_count = int(possible_series.astype(bool).sum())
                row["Possible"] = possible_count
                row["Possible rate"] = possible_count / evaluated if evaluated else None

        if has_match_type and match_type_values:
            type_counts = (
                group["match_type"].fillna("(missing)").astype(str).value_counts()
            )
            for raw_type in match_type_values:
                column_name = _display_match_type(raw_type)
                row[column_name] = int(type_counts.get(raw_type, 0))

        rows.append(row)

    if not rows:
        return ""

    dataset_df = pd.DataFrame(rows)

    sort_column = "Accuracy" if "Accuracy" in dataset_df.columns else None
    if sort_column is not None:
        dataset_df = dataset_df.sort_values(by=sort_column, ascending=False)

    formatters: dict[str, Any] = {
        "Evaluated": _as_int,
        "Correct": _as_int,
        "Accuracy": _as_percent,
    }
    if has_possible:
        formatters.update({"Possible": _as_int, "Possible rate": _as_percent})

    if has_match_type and match_type_values:
        for raw_type in match_type_values:
            column_name = _display_match_type(raw_type)
            formatters[column_name] = _as_int

    display_df = dataset_df.copy()
    for column, formatter in formatters.items():
        if column in display_df:
            display_df[column] = display_df[column].map(formatter)

    return _markdown(display_df)


def _build_top_error_sections(df: pd.DataFrame | None) -> list[str]:
    sections: list[str] = []
    if df is None:
        return sections
    resolved_col = None
    for candidate in ("resolved_keywords",):
        if candidate in df.columns:
            resolved_col = candidate
            break
        for column in df.columns:
            if str(column).lower().endswith(candidate):
                resolved_col = column
                break
        if resolved_col:
            break

    correct_col = None
    for column in df.columns:
        if str(column).lower().endswith("correct"):
            correct_col = column
            break
    if correct_col is None:
        return sections

    incorrect_mask = df[correct_col] == False  # noqa: E712
    incorrect_df = df[incorrect_mask]

    if incorrect_df.empty:
        return sections

    if resolved_col and resolved_col in incorrect_df.columns:
        series = incorrect_df[resolved_col].dropna()
        if not series.empty:
            exploded = series.apply(
                lambda value: list(value)
                if isinstance(value, (list, tuple, set))
                else ([value] if isinstance(value, str) else [])
            ).explode()
        else:
            exploded = series
        top_wrong = exploded.fillna("(missing)").astype(str).value_counts()
        if not top_wrong.empty:
            top_wrong_df = (
                top_wrong.head(10).rename_axis("Keyword").reset_index(name="Count")
            )
            top_wrong_table = _markdown(top_wrong_df)
            if top_wrong_table:
                sections.extend(
                    ["### Top wrong predicted labels", top_wrong_table, ""]
                )

    gold_col = None
    for column in incorrect_df.columns:
        if str(column).lower().endswith("gold_labels"):
            gold_col = column
            break
    if gold_col:
        gold_series = incorrect_df[gold_col].dropna()
        if not gold_series.empty:
            exploded = gold_series.apply(
                lambda value: list(value)
                if isinstance(value, (list, tuple, set))
                else ([value] if isinstance(value, str) else [])
            ).explode()
            exploded = exploded.dropna()
            if not exploded.empty:
                top_gold = exploded.astype(str).value_counts()
                if not top_gold.empty:
                    top_gold_df = (
                        top_gold.head(10)
                        .rename_axis("Keyword")
                        .reset_index(name="Count")
                    )
                    top_gold_table = _markdown(top_gold_df)
                    if top_gold_table:
                        sections.extend(
                            [
                                "### Top wrong gold labels",
                                top_gold_table,
                                "",
                            ]
                        )

    return sections


def report_results(
    df: pd.DataFrame | None,
    metrics: dict[str, Any] | None = None,
    *,
    display_columns: Sequence[str] | None = None,
    dataset_column: str | None = "dataset",
    extra_messages: Iterable[str] | None = None,
    output_path: str
    | Path
    | tuple[str | Path, str | Path | None]
    | None = None,
) -> None:
    """Print the evaluation results and derived metrics.

    When ``output_path`` is provided the Markdown summary generated by
    :func:`format_metrics` is written to the supplied path. Supplying a tuple of
    paths additionally writes the console-oriented plain-text output to the
    second location.
    """

    display_columns = (
        list(display_columns)
        if display_columns is not None
        else [
            "label",
            "name",
            "description",
            "gold_labels",
            "resolved_keywords",
            "correct",
            "match_type",
        ]
    )

    if df is not None and display_columns:
        lower_map = {str(col).lower(): col for col in df.columns}
        resolved: list[str] = []
        for column in display_columns:
            if column in df.columns:
                resolved.append(column)
                continue
            candidate = lower_map.get(str(column).lower())
            if candidate:
                resolved.append(candidate)
                continue
            for col in df.columns:
                if str(col).lower().endswith(str(column).lower()):
                    resolved.append(col)
                    break
        if resolved:
            display_columns = list(dict.fromkeys(resolved))

    text_sections: list[str] = []

    if df is not None:
        present_cols = [c for c in display_columns if c in df.columns]
        if present_cols:
            display_df = df[present_cols]
            print(display_df)
            text_sections.append(_dataframe_to_text(display_df))
        else:
            print(df)
            text_sections.append(_dataframe_to_text(df))

    possible_count, possible_rate = _possible_correct_summary(metrics)
    if possible_count is not None:
        rate_str = f" ({possible_rate * 100:.2f}%)" if possible_rate is not None else ""
        print(f"Possible correct under allowed: {possible_count:,}{rate_str}")
        text_sections.append(
            f"Possible correct under allowed: {possible_count:,}{rate_str}"
        )

    if extra_messages:
        for message in extra_messages:
            if message:
                print(message)
                text_sections.append(str(message))

    metrics_report = format_metrics(metrics, df, dataset_column=dataset_column)
    if metrics_report:
        print(metrics_report)
        text_sections.append(metrics_report)

    if output_path is not None:
        markdown_path: Path
        text_path: Path | None = None

        if isinstance(output_path, tuple):
            markdown_path = Path(output_path[0])
            second = output_path[1]
            text_path = Path(second) if second is not None else None
        else:
            markdown_path = Path(output_path)

        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        if markdown_path.exists() and markdown_path.is_dir():
            raise ValueError("output_path must be a file path, not a directory")
        markdown_content = (metrics_report or "").strip()
        markdown_path.write_text(
            (markdown_content + "\n") if markdown_content else "",
            encoding="utf-8",
        )

        if text_path is not None:
            text_path.parent.mkdir(parents=True, exist_ok=True)
            if text_path.exists() and text_path.is_dir():
                raise ValueError("plain text output path must be a file path")
            text_content = "\n\n".join(filter(None, text_sections)).strip()
            text_path.write_text(
                (text_content + "\n") if text_content else "",
                encoding="utf-8",
            )


def _possible_correct_summary(
    metrics: dict[str, Any] | None,
) -> tuple[int | None, float | None]:
    if not metrics:
        return None, None

    count = metrics.get("n_possible_correct_under_allowed")
    rate = metrics.get("possible_correct_under_allowed_rate")

    if count is None:
        return None, None

    try:
        count_int = int(count)
    except (TypeError, ValueError):
        return None, None

    rate_float: float | None = None
    if rate is not None:
        try:
            rate_float = float(rate)
        except (TypeError, ValueError):
            rate_float = None

    return count_int, rate_float
