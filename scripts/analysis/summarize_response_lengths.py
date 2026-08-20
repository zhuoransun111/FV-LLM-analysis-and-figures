#!/usr/bin/env python3
"""Recompute the cleaned response-length summary used for Table S3.

Each question-source response is counted once, irrespective of how many
evaluator records it has.  The input should therefore contain one stable
``response_length`` for every ``response_id``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=code_root / "data/analysis_dataset.csv",
    )
    parser.add_argument(
        "--source-codebook", type=Path,
        default=code_root / "data/source_codebook.csv",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=code_root / "results/descriptive/response_length_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    required = {"response_id", "group_code", "response_length"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input dataset is missing columns: {missing}")

    within_response_nunique = data.groupby("response_id")["response_length"].nunique(
        dropna=False
    )
    if not bool((within_response_nunique == 1).all()):
        bad = within_response_nunique[within_response_nunique != 1].index.tolist()
        raise ValueError(f"Inconsistent lengths within response_id: {bad[:10]}")

    # Each response_id has two distinct rating records (one per evaluator panel)
    # but only one response text and one length. Selecting one metadata row per
    # response here does not remove any allocation or rating from the analyses.
    responses = data.drop_duplicates("response_id").copy()
    codebook = pd.read_csv(args.source_codebook)
    responses = responses.merge(codebook, on="group_code", how="left", validate="many_to_one")
    if responses["response_source"].isna().any():
        raise ValueError("Source codebook does not cover every group_code")

    summary = (
        responses.groupby(["group_code", "response_source"], sort=True)["response_length"]
        .agg(
            n_responses="count",
            mean_characters="mean",
            sd_characters="std",
            median_characters="median",
            minimum_characters="min",
            maximum_characters="max",
        )
        .reset_index()
    )
    quartiles = (
        responses.groupby(["group_code", "response_source"])["response_length"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "Q1_characters", 0.75: "Q3_characters"})
        .reset_index()
    )
    summary = summary.merge(
        quartiles, on=["group_code", "response_source"], validate="one_to_one"
    )
    ordered = [
        "group_code", "response_source", "n_responses", "mean_characters",
        "sd_characters", "median_characters", "Q1_characters", "Q3_characters",
        "minimum_characters", "maximum_characters",
    ]
    summary = summary[ordered]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False, float_format="%.2f")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
