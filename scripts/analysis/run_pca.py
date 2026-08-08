#!/usr/bin/env python3
"""Recompute the panel-specific PCA reported in Table S2 and Fig. S1.

The calculation reproduces ``StandardScaler`` (population standard deviation,
``ddof=0``) followed by unrotated PCA on complete cases within each evaluator
panel.  It is implemented with NumPy SVD so the exact calculation does not
depend on scikit-learn.  Component signs are oriented deterministically because
PCA signs are otherwise arbitrary: PC1 has a negative sum of loadings and the
sixth-item loading on PC2 is negative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ITEMS = [
    ("Q1OEAE", "Correctness/reliability"),
    ("Q2OEAA", "Clarity"),
    ("Q3OEAA", "Completeness"),
    ("Q4OEOEOE", "Empathy"),
    ("Q5OEAA", "Fairness/non-misleadingness"),
    ("Q6OEAEOEOE", "Perceived absence of bias"),
]


def cronbach_alpha(values: np.ndarray) -> float:
    item_variances = values.var(axis=0, ddof=1)
    total_variance = values.sum(axis=1).var(ddof=1)
    item_count = values.shape[1]
    return float(
        (item_count / (item_count - 1))
        * (1.0 - item_variances.sum() / total_variance)
    )


def panel_pca(frame: pd.DataFrame, panel: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [column for column, _ in ITEMS]
    complete = frame.loc[frame["panel"] == panel, columns].dropna()
    values = complete.to_numpy(dtype=float)
    if values.shape[1] != 6 or values.shape[0] < 2:
        raise ValueError(f"{panel}: insufficient complete data for six-item PCA")

    scaled = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    _, singular_values, right_vectors = np.linalg.svd(scaled, full_matrices=False)
    loadings = right_vectors[:2].T.copy()

    # Deterministic orientation matching the archived Table S2/Fig. S1.
    if loadings[:, 0].sum() > 0:
        loadings[:, 0] *= -1
    if loadings[5, 1] > 0:
        loadings[:, 1] *= -1

    eigenvalues = singular_values**2 / (len(scaled) - 1)
    explained = eigenvalues / eigenvalues.sum()
    rows = pd.DataFrame(
        {
            "panel": panel,
            "item_code": columns,
            "rating_item": [label for _, label in ITEMS],
            "PC1_loading": loadings[:, 0],
            "PC2_loading": loadings[:, 1],
        }
    )
    summary = {
        "panel": panel,
        "input_rows": int((frame["panel"] == panel).sum()),
        "complete_case_rows": int(len(complete)),
        "rows_excluded_for_any_missing_item": int(
            (frame["panel"] == panel).sum() - len(complete)
        ),
        "standardization": "item-wise mean 0 and population SD 1 (ddof=0)",
        "rotation": "none",
        "PC1_explained_variance_percent": float(explained[0] * 100),
        "PC2_explained_variance_percent": float(explained[1] * 100),
        "cronbach_alpha_complete_cases": cronbach_alpha(values),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=code_root / "data/analysis_dataset.csv",
    )
    parser.add_argument(
        "--loadings-csv",
        type=Path,
        default=code_root / "results/descriptive/pca_loadings.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=code_root / "results/descriptive/pca_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    required = {"panel", *(column for column, _ in ITEMS)}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input dataset is missing columns: {missing}")

    output_rows = []
    summaries = []
    for panel in ("Expert", "Parent"):
        rows, summary = panel_pca(data, panel)
        output_rows.append(rows)
        summaries.append(summary)

    loadings = pd.concat(output_rows, ignore_index=True)
    args.loadings_csv.parent.mkdir(parents=True, exist_ok=True)
    loadings.to_csv(args.loadings_csv, index=False, float_format="%.8f")
    args.summary_json.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(loadings.to_string(index=False))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
