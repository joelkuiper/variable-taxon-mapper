#!/usr/bin/env python3
"""Stratified sampler for variable results tables.

The script draws a proportional sample from a results table that contains
multiple datasets. Sampling is stratified by the dataset column so that the
output represents the mix of the full table (e.g., ~20k rows).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Load a CSV/Parquet/Feather table."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".feather", ".ft"):
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file extension for {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV/Parquet/Feather based on suffix."""
    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    elif ext in (".feather", ".ft"):
        df.to_feather(path)
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def allocate_samples_per_dataset(group_sizes: pd.Series, target_total: int) -> Dict[str, int]:
    """Allocate how many rows to sample per dataset.

    Uses the largest remainder method so allocations are proportional to the
    dataset's share of the full table while summing to the target_total. If the
    target exceeds the available rows, the allocation defaults to the full
    dataset sizes.
    """

    total_rows = int(group_sizes.sum())
    if total_rows == 0:
        return {}

    capped_target = min(target_total, total_rows)
    if capped_target <= 0:
        return {}

    num_groups = len(group_sizes)
    if capped_target < num_groups:
        raise ValueError(
            "Target sample size is smaller than the number of datasets; "
            "cannot include at least one row from each dataset."
        )

    # Reserve one row per dataset to guarantee coverage.
    base_allocation = pd.Series(np.ones(num_groups, dtype=int), index=group_sizes.index)
    remaining_target = capped_target - num_groups

    available_sizes = group_sizes - base_allocation
    available_sizes = available_sizes.clip(lower=0)
    remaining_capacity = int(available_sizes.sum())

    if remaining_target == 0 or remaining_capacity == 0:
        allocation = base_allocation.copy()
        remainders = pd.Series(0.0, index=group_sizes.index)
    else:
        fractional = (available_sizes / remaining_capacity) * remaining_target
        extra_base = np.floor(fractional).astype(int)
        remainders = fractional - extra_base
        allocation = base_allocation + extra_base

    remaining = capped_target - int(allocation.sum())

    if remaining > 0:
        # Distribute leftover counts by largest remainder while respecting
        # dataset size caps.
        remainder_order = remainders.sort_values(ascending=False).index.tolist()
        idx = 0
        while remaining > 0 and remainder_order:
            ds = remainder_order[idx % len(remainder_order)]
            if allocation[ds] < group_sizes[ds]:
                allocation[ds] += 1
                remaining -= 1
            idx += 1
            if idx >= 10 * len(remainder_order):
                # Avoid infinite loops if all groups are saturated.
                break

    # Clip to dataset sizes in case rounding pushed us over.
    for ds, size in group_sizes.items():
        allocation[ds] = min(int(allocation.get(ds, 0)), int(size))

    # Ensure the final sum does not exceed capped_target by trimming smallest
    # allocations if needed.
    current_total = int(allocation.sum())
    if current_total > capped_target:
        overshoot = current_total - capped_target
        # Sort by smallest contribution first for trimming.
        for ds in sorted(allocation.index, key=lambda x: allocation[x]):
            if overshoot <= 0:
                break
            trim = min(overshoot, allocation[ds])
            allocation[ds] -= trim
            overshoot -= trim

    return {k: v for k, v in allocation.items() if v > 0}


def stratified_sample(df: pd.DataFrame, dataset_column: str, target_total: int, seed: int | None) -> pd.DataFrame:
    """Return a stratified sample of df based on dataset_column."""

    if dataset_column not in df.columns:
        raise ValueError(f"Column '{dataset_column}' not found in input table")

    group_sizes = df[dataset_column].value_counts()
    allocations = allocate_samples_per_dataset(group_sizes, target_total)

    samples = []
    for dataset_name, n_rows in allocations.items():
        group_df = df[df[dataset_column] == dataset_name]
        if n_rows >= len(group_df):
            samples.append(group_df)
        else:
            samples.append(group_df.sample(n=n_rows, random_state=seed))

    return pd.concat(samples, ignore_index=True) if samples else df.head(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified sampler for variable results tables")
    parser.add_argument("input", type=Path, help="Input results file (CSV/Parquet/Feather)")
    parser.add_argument("output", type=Path, help="Where to write the stratified sample")
    parser.add_argument(
        "--dataset-column",
        default="dataset",
        help="Column that identifies the dataset grouping for stratification",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=2000,
        help="Target total number of rows to sample (defaults to 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[sample_results] Loading table from: {args.input}")
    df = read_table(args.input)
    print(f"[sample_results] Loaded {len(df)} rows across {df[args.dataset_column].nunique()} datasets")

    sampled = stratified_sample(df, args.dataset_column, args.target, args.seed)
    print(f"[sample_results] Allocated {len(sampled)} rows for output")

    write_table(sampled, args.output)
    print(f"[sample_results] Wrote sample to: {args.output}")


if __name__ == "__main__":
    main()
