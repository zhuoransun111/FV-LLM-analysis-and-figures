#!/usr/bin/env python3
"""Validate the frozen analysis-ready dataset before model fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTCOMES = ["Q1OEAE", "Q2OEAA", "Q3OEAA", "Q4OEOEOE", "Q5OEAA", "Q6OEAEOEOE"]
REQUIRED = {
    "question_id", "group", "panel", "rater", *OUTCOMES,
    "response_length", "difficulty", "response_id", "group_code", "length_100",
}
KEY = ["panel", "question_id", "group_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    missing_columns = sorted(REQUIRED - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if len(data) != 2600:
        raise ValueError(f"Expected 2,600 rows; found {len(data):,}")
    if data[OUTCOMES].isna().any().any():
        raise ValueError("Outcome ratings contain missing values")
    if not data[OUTCOMES].apply(lambda column: column.between(1, 5).all()).all():
        raise ValueError("Outcome ratings must all be in the range 1–5")

    data["group_code"] = data["group_code"].astype(int)
    expected = pd.MultiIndex.from_product(
        [["Expert", "Parent"], range(1, 101), range(1, 14)], names=KEY
    )
    observed = pd.MultiIndex.from_frame(data[KEY])
    if observed.has_duplicates:
        raise ValueError("Panel–question–source keys are not unique")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))[:20]
        unexpected = sorted(set(observed) - set(expected))[:20]
        raise ValueError(f"Incomplete allocation grid; missing={missing}, unexpected={unexpected}")

    response_meta = data.groupby("response_id")[["question_id", "group_code", "response_length", "difficulty"]].nunique(dropna=False)
    if (response_meta > 1).any().any():
        raise ValueError("Response metadata are inconsistent across evaluator panels")
    expected_length_100 = data["response_length"] / 100
    if not (data["length_100"].sub(expected_length_100).abs() < 1e-12).all():
        raise ValueError("length_100 is inconsistent with response_length")

    print(
        "Validation passed: 2,600 complete ratings; 1,300 per panel; "
        "all 2 × 100 × 13 panel–question–source keys present exactly once."
    )


if __name__ == "__main__":
    main()
