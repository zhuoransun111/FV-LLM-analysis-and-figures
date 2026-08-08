#!/usr/bin/env python3
"""Summarize within-rater standardized scores for Figs. 1, 2 and S2."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ITEMS = {
    "Q1OEAE": "Correctness/reliability",
    "Q2OEAA": "Clarity",
    "Q3OEAA": "Completeness",
    "Q4OEOEOE": "Empathy",
    "Q5OEAA": "Fairness/non-misleadingness",
    "Q6OEAEOEOE": "Perceived absence of bias",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/analysis_dataset.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "results/descriptive/standardized_scores.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    required = {"panel", "rater", "group_code", *ITEMS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input dataset is missing columns: {missing}")

    data["group_code"] = data["group_code"].astype(str)
    rows: list[pd.DataFrame] = []
    for column, label in ITEMS.items():
        subset = data[["panel", "rater", "group_code", column]].dropna().copy()
        subset["z_score"] = subset.groupby(["panel", "rater"])[column].transform(
            lambda values: (values - values.mean()) / values.std(ddof=1)
        )
        summary = (
            subset.groupby(["panel", "group_code"], sort=True)["z_score"]
            .agg(mean_z="mean", sem_z="sem", n_ratings="count")
            .reset_index()
        )
        summary.insert(2, "rating_item", label)
        rows.append(summary)

    output = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False, float_format="%.10f")
    print(args.out)


if __name__ == "__main__":
    main()

