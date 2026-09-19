"""Summarize changes after adding human response-provider clustering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ["dimension", "panel", "group_code"]


def compare(base_path: Path, provider_path: Path) -> dict:
    base = pd.read_csv(base_path)
    provider = pd.read_csv(provider_path)
    merged = base.merge(provider, on=KEYS, suffixes=("_base", "_provider"))
    if len(merged) != 24:
        raise ValueError(f"Expected 24 matched contrasts; found {len(merged)}.")
    beta_change = merged["beta_provider"] - merged["beta_base"]
    significant_base = merged["p_holm_base"].lt(0.05)
    significant_provider = merged["p_holm_provider"].lt(0.05)
    return {
        "contrasts": int(len(merged)),
        "maximum_absolute_beta_change": float(beta_change.abs().max()),
        "holm_significance_changes": int(
            significant_base.ne(significant_provider).sum()
        ),
        "holm_significant_base": int(significant_base.sum()),
        "holm_significant_provider_clustered": int(significant_provider.sum()),
        "provider_clustered_confidence_intervals_excluding_zero": int(
            (merged["ci_low_provider"].gt(0) | merged["ci_high_provider"].lt(0)).sum()
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "results/provider_clustered/sensitivity_summary.json",
    )
    args = parser.parse_args()
    summary = {
        "primary": compare(
            root / "results/primary/primary_unadjusted_contrasts.csv",
            root
            / "results/provider_clustered/primary_provider_clustered_contrasts.csv",
        ),
        "length_adjusted": compare(
            root / "results/length_adjusted/length_adjusted_contrasts.csv",
            root
            / "results/provider_clustered/length_adjusted_provider_clustered_contrasts.csv",
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
