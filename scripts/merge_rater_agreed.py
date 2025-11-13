#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path
from typing import List

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif ext in (".feather", ".ft"):
        return pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def write_table(df: pd.DataFrame, path: Path, csv_sep_keywords: str):
    ext = path.suffix.lower()
    if ext == ".csv":
        df_out = df.copy()
        for col in df_out.columns:
            if df_out[col].apply(lambda x: isinstance(x, (list, tuple, set))).any():
                df_out[col] = df_out[col].apply(
                    lambda v: csv_sep_keywords.join(map(str, v))
                    if isinstance(v, (list, tuple, set))
                    else v
                )
                df_out.to_csv(path, index=False)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    elif ext in (".feather", ".ft"):
        df.to_feather(path)
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def keyify(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ("dataset", "label", "name"):
        if col in df.columns:
            parts.append(df[col].astype(str).str.strip())
        else:
            parts.append(pd.Series([""], index=df.index))
    return parts[0] + "||" + parts[1] + "||" + parts[2]


def parse_keywords_cell(v, csv_sep_keywords: str) -> List[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == ""):
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()

    try:
        lit = ast.literal_eval(s)
        if isinstance(lit, (list, tuple, set)):
            return [str(x).strip() for x in lit if str(x).strip()]
    except Exception:
        pass

    try:
        j = json.loads(s)
        if isinstance(j, list):
            return [str(x).strip() for x in j if str(x).strip()]
    except Exception:
        pass

    if csv_sep_keywords in s:
        parts = [p.strip() for p in s.split(csv_sep_keywords)]
        return [p for p in parts if p]
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return [p for p in parts if p]

    return [s] if s else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variables",
        required=True,
        type=Path,
        help="Path to variables table (csv/parquet/feather)",
    )
    ap.add_argument(
        "--review",
        dest="reviews",
        required=True,
        action="append",
        type=Path,
        help="Path to a reviewer CSV (can be repeated)",
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for updated variables table",
    )
    ap.add_argument(
        "--keywords-column",
        default="keywords",
        help="Name of the keywords column to update/create",
    )
    # Accept both spellings: --csv-keywords-sep and --csv-keyword-sep
    ap.add_argument(
        "--csv-keywords-sep",
        "--csv-keyword-sep",
        dest="csv_keywords_sep",
        default="|",
        help="Delimiter for CSV serialization of keywords lists",
    )
    args = ap.parse_args()

    if len(args.reviews) < 2:
        ap.error("At least two --review files are required to define agreement.")

    print(f"[merge_rater_agreed] Loading variables from: {args.variables}")
    vars_df = read_table(args.variables)

    print(f"[merge_rater_agreed] Loading {len(args.reviews)} review files:")
    review_dfs = []
    for p in args.reviews:
        print(f"  - {p}")
        review_dfs.append(pd.read_csv(p))

    accepted_dfs = [
        df[df["decision"].astype(str).str.lower() == "accept"].copy()
        for df in review_dfs
    ]
    accepted_counts = [len(df) for df in accepted_dfs]
    print(f"[merge_rater_agreed] Accepted rows per reviewer: {accepted_counts}")

    can_match_on_index = (
        "row_index" in vars_df.columns
        and all("row_index" in df.columns for df in accepted_dfs)
    )
    print(f"[merge_rater_agreed] Matching strategy: "
          f"{'row_index' if can_match_on_index else 'composite key (dataset|label|name)'}")

    if args.keywords_column not in vars_df.columns:
        vars_df[args.keywords_column] = ""

    updated_rows = 0  # track how many rows actually change

    if can_match_on_index:
        index_sets = [set(df["row_index"]) for df in accepted_dfs]
        agreed_indices = set.intersection(*index_sets) if index_sets else set()
        print(f"[merge_rater_agreed] Agreed row_index count: {len(agreed_indices)}")

        index_to_labels = {}
        for df in accepted_dfs:
            for idx, labels in (
                    df[df["row_index"].isin(agreed_indices)]
                    .groupby("row_index")["resolved_label"]
            ):
                labels_set = index_to_labels.setdefault(idx, set())
                labels_set.update(map(str, labels))

        for idx in list(index_to_labels.keys()):
            index_to_labels[idx] = sorted(index_to_labels[idx])

        def add_labels_row(row):
            nonlocal updated_rows
            idx = row.get("row_index", None)
            if idx in index_to_labels:
                existing = parse_keywords_cell(row[args.keywords_column], args.csv_keywords_sep)
                new = list(dict.fromkeys(existing + index_to_labels[idx]))
                if new != existing:
                    updated_rows += 1
                return new
            return row[args.keywords_column]

        vars_df[args.keywords_column] = vars_df.apply(add_labels_row, axis=1)

    else:
        for df in accepted_dfs:
            df["__key"] = keyify(df)

        key_sets = [set(df["__key"]) for df in accepted_dfs]
        agreed_keys = set.intersection(*key_sets) if key_sets else set()
        print(f"[merge_rater_agreed] Agreed composite key count: {len(agreed_keys)}")

        key_to_labels = {}
        for df in accepted_dfs:
            for k, labels in (
                    df[df["__key"].isin(agreed_keys)]
                    .groupby("__key")["resolved_label"]
            ):
                labels_set = key_to_labels.setdefault(k, set())
                labels_set.update(map(str, labels))

        for k in list(key_to_labels.keys()):
            key_to_labels[k] = sorted(key_to_labels[k])

        vars_df["__key"] = keyify(vars_df)

        def add_labels_key(row):
            nonlocal updated_rows
            k = row["__key"]
            if k in key_to_labels:
                existing = parse_keywords_cell(row[args.keywords_column], args.csv_keywords_sep)
                new = list(dict.fromkeys(existing + key_to_labels[k]))
                if new != existing:
                    updated_rows += 1
                return new
            return row[args.keywords_column]

        vars_df[args.keywords_column] = vars_df.apply(add_labels_key, axis=1)
        vars_df.drop(columns=["__key"], inplace=True, errors="ignore")

    write_table(vars_df, args.output, csv_sep_keywords=args.csv_keywords_sep)
    print(f"[merge_rater_agreed] Updated keywords written to: {args.output}")
    print(f"[merge_rater_agreed] Rows updated: {updated_rows}")


if __name__ == "__main__":
    main()
