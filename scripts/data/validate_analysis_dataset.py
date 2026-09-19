#!/usr/bin/env python3
"""Validate the frozen analysis-ready dataset before model fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OUTCOMES = ["Q1OEAE", "Q2OEAA", "Q3OEAA", "Q4OEOEOE", "Q5OEAA", "Q6OEAEOEOE"]
REQUIRED = {
    "question_id", "group", "panel", "rater", *OUTCOMES,
    "response_length", "difficulty", "response_id", "group_code", "length_100",
}
KEY = ["panel", "question_id", "group_code"]
HUMAN_SOURCE_CODES = {10, 11, 12, 13}


def validate(data: pd.DataFrame) -> dict:
    missing_columns = sorted(REQUIRED - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if len(data) != 2600:
        raise ValueError(f"Expected 2,600 rows; found {len(data):,}")
    if data[OUTCOMES].isna().any().any():
        raise ValueError("Outcome ratings contain missing values")
    if not data[OUTCOMES].apply(lambda column: column.between(1, 5).all()).all():
        raise ValueError("Outcome ratings must all be in the range 1-5")

    checked = data.copy()
    checked["group_code"] = checked["group_code"].astype(int)
    expected = pd.MultiIndex.from_product(
        [["Expert", "Parent"], range(1, 101), range(1, 14)], names=KEY
    )
    observed = pd.MultiIndex.from_frame(checked[KEY])
    if observed.has_duplicates:
        raise ValueError("Panel-question-source keys are not unique")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))[:20]
        unexpected = sorted(set(observed) - set(expected))[:20]
        raise ValueError(
            f"Incomplete allocation grid; missing={missing}, unexpected={unexpected}"
        )

    response_meta = checked.groupby("response_id")[
        ["question_id", "group_code", "response_length", "difficulty"]
    ].nunique(dropna=False)
    if (response_meta > 1).any().any():
        raise ValueError("Response metadata are inconsistent across evaluator panels")
    expected_length_100 = checked["response_length"] / 100
    if not (checked["length_100"].sub(expected_length_100).abs() < 1e-12).all():
        raise ValueError("length_100 is inconsistent with response_length")

    report = {
        "rows": int(len(checked)),
        "unique_panel_question_source_keys": int(len(observed.unique())),
        "duplicate_key_rows": 0,
        "missing_expected_keys": 0,
        "missing_outcome_values": {
            name: int(checked[name].isna().sum()) for name in OUTCOMES
        },
        "panels": sorted(checked["panel"].unique().tolist()),
        "questions": int(checked["question_id"].nunique()),
        "response_sources": int(checked["group_code"].nunique()),
    }

    if "response_provider" in checked.columns:
        human = checked[checked["group_code"].isin(HUMAN_SOURCE_CODES)].copy()
        if human["response_provider"].isna().any():
            raise ValueError("Human-source rows contain missing response_provider values")
        human["response_provider"] = human["response_provider"].astype(str).str.strip()
        if human["response_provider"].isin({"", "nan", "none", "null"}).any():
            raise ValueError("Human-source rows contain blank response_provider values")

        providers_per_source = human.groupby("group_code")["response_provider"].nunique()
        if not providers_per_source.eq(10).all():
            raise ValueError(
                "Expected 10 anonymous response providers for each human source; "
                f"found {providers_per_source.to_dict()}"
            )
        provider_source_counts = human.groupby("response_provider")["group_code"].nunique()
        if not provider_source_counts.eq(1).all() or human["response_provider"].nunique() != 40:
            raise ValueError(
                "Expected 40 source-specific anonymous response providers, with no "
                "provider code reused across human sources"
            )

        records_per_provider = human.groupby("response_provider").size()
        questions_per_provider = human.groupby("response_provider")["question_id"].nunique()
        provider_question = human.groupby(
            ["response_provider", "question_id"]
        ).agg(records=("panel", "size"), panels=("panel", "nunique"))
        if not (
            records_per_provider.eq(20).all()
            and questions_per_provider.eq(10).all()
            and provider_question["records"].eq(2).all()
            and provider_question["panels"].eq(2).all()
        ):
            raise ValueError(
                "Each anonymous provider must contribute 10 distinct answers, each with "
                "one expert-panel and one parent-panel rating record"
            )
        report["anonymous_human_response_providers"] = int(
            human["response_provider"].nunique()
        )
        report["records_per_human_response_provider"] = 20
        report["questions_per_human_response_provider"] = 10

    return report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=root / "results/data_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(pd.read_csv(args.data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
