"""Create the duplicate-assignment sensitivity dataset and audit summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY = ["panel", "question_id", "group_code"]
EXPECTED_COLUMNS = {
    "question_id",
    "panel",
    "rater",
    "group_code",
    "difficulty",
    "response_id",
    "length_100",
    "Q1OEAE",
    "Q2OEAA",
    "Q3OEAA",
    "Q4OEOEOE",
    "Q5OEAA",
    "Q6OEAEOEOE",
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/analysis_dataset.csv")
    parser.add_argument(
        "--out", type=Path, default=root / "data/analysis_dataset_deduplicated.csv"
    )
    parser.add_argument(
        "--audit", type=Path, default=root / "results/sensitivity/duplicate_audit.json"
    )
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    missing_columns = sorted(EXPECTED_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    duplicate_mask = data.duplicated(KEY, keep=False)
    extra_mask = data.duplicated(KEY, keep="first")
    deduplicated = data.loc[~extra_mask].copy()
    panel_extras = (
        data.loc[extra_mask, "panel"].value_counts().sort_index().to_dict()
    )
    audit = {
        "input_rows": int(len(data)),
        "output_rows": int(len(deduplicated)),
        "duplicate_excess_rows": int(extra_mask.sum()),
        "rows_participating_in_duplicate_keys": int(duplicate_mask.sum()),
        "duplicate_excess_rows_by_panel": {
            str(key): int(value) for key, value in panel_extras.items()
        },
        "missing_clarity_values": int(data["Q2OEAA"].isna().sum()),
        "duplicate_key": KEY,
        "retention_rule": (
            "Retain the first record for each panel-question-source key in the "
            "archived row order; remove subsequent records only for sensitivity analysis."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    deduplicated.to_csv(args.out, index=False)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
