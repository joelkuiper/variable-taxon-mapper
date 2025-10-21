from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd

from config import AppConfig, load_config
from src.embedding import (
    Embedder,
    build_hnsw_index,
    build_taxonomy_embeddings_composed,
)
from src.evaluate import ProgressHook, run_label_benchmark
from src.taxonomy import (
    build_gloss_map,
    build_name_maps_from_graph,
    build_taxonomy_graph,
)
from src.utils import set_global_seed


def _prepare_keywords(
    keywords: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    required_columns = {"name", "parent", "order"}
    missing = sorted(required_columns - set(keywords.columns))
    if missing:
        raise KeyError(f"Keywords data missing required columns: {missing}")

    summary_df = keywords if "definition_summary" in keywords.columns else None

    return keywords, summary_df


def run_pipeline(
    config: AppConfig,
    *,
    base_path: Path | None = None,
    variables_csv: Path | None = None,
    evaluate: bool = True,
    progress_hook: ProgressHook | None = None,
) -> Tuple[pd.DataFrame, dict[str, object]]:
    set_global_seed(config.seed)

    variables_default, keywords_path = config.data.to_paths(base_path)
    variables_path = variables_csv or variables_default

    parallel_cfg = config.parallelism
    print(
        "[pipeline] concurrency settings: "
        f"pruning_workers={parallel_cfg.pruning_workers}, "
        f"pruning_batch={parallel_cfg.pruning_batch_size}"
    )

    variables = pd.read_csv(variables_path, low_memory=False)
    keywords_raw = pd.read_csv(keywords_path)

    keywords, summary_df = _prepare_keywords(keywords_raw)

    G = build_taxonomy_graph(
        keywords,
        name_col="name",
        parent_col="parent",
        order_col="order",
    )
    name_to_id, name_to_path = build_name_maps_from_graph(G)

    embedder = Embedder(**config.embedder.to_kwargs())

    taxonomy_kwargs = config.taxonomy_embeddings.to_kwargs()
    summary_source = summary_df if summary_df is not None else None
    tax_names, tax_embs = build_taxonomy_embeddings_composed(
        G,
        embedder,
        summaries=summary_source,
        **taxonomy_kwargs,
    )

    hnsw_index = build_hnsw_index(tax_embs, **config.hnsw.to_kwargs())
    gloss_map = build_gloss_map(summary_source)

    df, metrics = run_label_benchmark(
        variables,
        keywords,
        G=G,
        embedder=embedder,
        tax_names=tax_names,
        tax_embs_unit=tax_embs,
        hnsw_index=hnsw_index,
        name_to_id=name_to_id,
        name_to_path=name_to_path,
        gloss_map=gloss_map,
        eval_config=config.evaluation,
        pruning_config=config.pruning,
        llm_config=config.llm,
        parallel_config=config.parallelism,
        http_config=config.http,
        evaluate=evaluate,
        progress_hook=progress_hook,
    )

    return df, metrics


def format_metrics(metrics: dict[str, Any], df: pd.DataFrame | None = None) -> str:
    def humanize(label: str) -> str:
        return label.replace("_", " ").replace("-", " ").title()

    def as_percent(value: Any) -> str:
        if value is None:
            return "–"
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return "–"
        except TypeError:
            pass
        return f"{float(value) * 100:.2f}%"

    def as_int(value: Any) -> str:
        if value is None:
            return "–"
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return "–"
        except TypeError:
            pass
        return f"{int(value):,}"

    def as_float(value: Any, digits: int = 4) -> str:
        if value is None:
            return "–"
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return "–"
        except TypeError:
            pass
        return f"{float(value):.{digits}f}"

    def markdown(df_obj: pd.DataFrame, *, index: bool = False) -> str:
        if df_obj.empty:
            return ""
        return df_obj.to_markdown(index=index, tablefmt="rounded_grid")

    metrics = metrics.copy()
    sections: list[str] = ["## Evaluation Metrics", ""]

    summary_fields: list[tuple[str, str, str]] = [
        ("Rows after dedupe", "n_total_rows_after_dedupe", "int"),
        ("Rows with keywords", "n_with_any_keyword", "int"),
        ("Eligible rows", "n_eligible", "int"),
        ("Excluded (not in taxonomy)", "n_excluded_not_in_taxonomy", "int"),
        ("Evaluated rows", "n_evaluated", "int"),
        ("Predictions correct", "n_correct", "int"),
        ("Predictions wrong", "n_unmatched", "int"),
        ("Accuracy (any match)", "label_accuracy_any_match", "pct"),
        ("Accuracy (exact only)", "label_accuracy_exact_only", "pct"),
        ("Accuracy (ancestor only)", "label_accuracy_ancestor_only", "pct"),
        ("Accuracy (descendant only)", "label_accuracy_descendant_only", "pct"),
        ("Errors", "n_errors", "int"),
    ]

    summary_rows = []
    for label, field, kind in summary_fields:
        if field not in metrics:
            continue
        value = metrics[field]
        if kind == "int":
            formatted = as_int(value)
        elif kind == "pct":
            formatted = as_percent(value)
        else:
            formatted = as_float(value)
        summary_rows.append({"Metric": label, "Value": formatted})

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_table = markdown(summary_df)
        if summary_table:
            sections.extend(["### Summary", summary_table, ""])

    counts = dict(metrics.get("match_type_counts", {}) or {})
    rates = dict(metrics.get("match_type_rates", {}) or {})
    if counts or rates:
        rows: list[dict[str, str]] = []
        for key in sorted({*counts.keys(), *rates.keys()}):
            display_key = "wrong" if key == "none" else key
            rows.append(
                {
                    "Match type": humanize(display_key),
                    "Count": as_int(counts.get(key)) if key in counts else "–",
                    "Rate": as_percent(rates.get(key)) if key in rates else "–",
                }
            )
        match_df = pd.DataFrame(rows)
        match_table = markdown(match_df)
        if match_table:
            sections.extend(["### Match types", match_table, ""])

    if df is not None and {"dataset", "match_strategy", "correct"}.issubset(df.columns):
        dataset_frame = df[["dataset", "match_strategy", "correct"]].dropna(
            subset=["dataset"]
        )
        if not dataset_frame.empty:
            dataset_frame = dataset_frame.copy()
            dataset_frame["dataset"] = dataset_frame["dataset"].astype(str)
            dataset_frame["match_strategy"] = (
                dataset_frame["match_strategy"].fillna("unknown").astype(str)
            )
            dataset_frame["correct"] = dataset_frame["correct"].astype(float)
            pivot = dataset_frame.pivot_table(
                index="dataset",
                columns="match_strategy",
                values="correct",
                aggfunc="mean",
            )
            if not pivot.empty:
                pivot = pivot.sort_index()
                pivot = pivot[[col for col in sorted(pivot.columns)]]
                pivot = pivot.applymap(lambda v: as_percent(v) if pd.notna(v) else "–")
                pivot.index.name = "Dataset"
                pivot.columns = [humanize(str(col)) for col in pivot.columns]
                dataset_table = markdown(pivot, index=True)
                if dataset_table:
                    sections.extend(
                        [
                            "### Accuracy by dataset and strategy",
                            dataset_table,
                            "",
                        ]
                    )

    other_raw = metrics.get("other_metrics")
    other = dict(other_raw) if isinstance(other_raw, dict) else {}

    hier_keys = [k for k in other if k.startswith("hierarchical_distance_")]
    if hier_keys:
        hier_rows: list[dict[str, str]] = []
        for key in sorted(hier_keys):
            value = other.pop(key)
            label = key.replace("hierarchical_distance_", "").replace("_", " ")
            label = label[0].upper() + label[1:] if label else label
            if key.endswith("_rate"):
                formatted = as_percent(value)
            elif key.endswith("_count"):
                formatted = as_int(value)
            else:
                formatted = as_float(value)
            hier_rows.append({"Hierarchical distance": label, "Value": formatted})
        hier_df = pd.DataFrame(hier_rows)
        hier_table = markdown(hier_df)
        if hier_table:
            sections.extend(["### Hierarchical distance metrics", hier_table, ""])

    strategy_perf = other.pop("match_strategy_performance", None)
    if isinstance(strategy_perf, dict):
        strategy_share = other.pop("match_strategy_correct_share", {}) or {}
        strategy_volume = other.pop("match_strategy_volume", {}) or {}
        strategy_rows: list[dict[str, str]] = []
        for key in sorted(strategy_perf):
            perf = strategy_perf[key] or {}
            strategy_rows.append(
                {
                    "Strategy": humanize(key),
                    "Volume": as_int(perf.get("n", strategy_volume.get(key))),
                    "Accuracy": as_percent(perf.get("accuracy")),
                    "Correct": as_int(perf.get("n_correct")),
                    "Correct share": as_percent(strategy_share.get(key)),
                }
            )
        strategy_df = pd.DataFrame(strategy_rows)
        strategy_table = markdown(strategy_df)
        if strategy_table:
            sections.extend(["### Match strategy performance", strategy_table, ""])

    additional_items: list[tuple[str, Any]] = []
    for key, value in sorted(other.items()):
        if isinstance(value, dict):
            formatted = json.dumps(value, indent=2, sort_keys=True)
            additional_items.append((humanize(key), f"`{formatted}`"))
        else:
            kind = (
                "pct"
                if str(key).endswith("_rate")
                else "int"
                if str(key).startswith("n_")
                else "float"
            )
            if kind == "pct":
                formatted = as_percent(value)
            elif kind == "int":
                formatted = as_int(value)
            else:
                formatted = as_float(value)
            additional_items.append((humanize(key), formatted))

    if additional_items:
        rows = [{"Metric": label, "Value": str(val)} for label, val in additional_items]
        additional_df = pd.DataFrame(rows)
        additional_table = markdown(additional_df)
        sections.extend(["### Additional metrics", additional_table, ""])

    return "\n".join(part for part in sections if part).strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the variable taxonomy mapper")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the TOML configuration file controlling the run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    base_path = config_path.parent
    config = load_config(config_path)
    set_global_seed(config.seed)
    variables_path, _ = config.data.to_paths(base_path)
    df, metrics = run_pipeline(
        config,
        base_path=base_path,
        variables_csv=variables_path,
    )

    display_cols = [
        "label",
        "name",
        "description",
        "gold_labels",
        "resolved_label",
        "correct",
        "match_type",
    ]
    present_cols = [c for c in display_cols if c in df.columns]
    if present_cols:
        print(df[present_cols])
    else:
        print(df)
    results_path = config.evaluation.resolve_results_path(
        base_path=base_path,
        variables_path=variables_path,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    metrics_report = format_metrics(metrics, df=df)
    print(metrics_report)


if __name__ == "__main__":
    main()
