"""Fit the primary mixed-effects models reported in the manuscript."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


OUTCOMES = {
    "Correctness/reliability": "Q1OEAE",
    "Clarity": "Q2OEAA",
    "Completeness": "Q3OEAA",
    "Empathy": "Q4OEOEOE",
    "Fair and non-misleading": "Q5OEAA",
    "Perceived absence of bias": "Q6OEAEOEOE",
}

SOURCE_NAMES = {"2": "RAG", "3": "RAG+Focus"}


def linear_contrast(fit, weights: dict[str, float]) -> dict[str, float]:
    names = list(fit.fe_params.index)
    vector = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    estimate = float(vector @ fit.fe_params.to_numpy())
    covariance = fit.cov_params().loc[names, names].to_numpy()
    se = float(np.sqrt(vector @ covariance @ vector))
    z = estimate / se
    from scipy.stats import norm

    p = float(2 * norm.sf(abs(z)))
    return {
        "beta": estimate,
        "se": se,
        "ci_low": estimate - 1.96 * se,
        "ci_high": estimate + 1.96 * se,
        "p_raw": p,
    }


def fit_models(data: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    contrasts: list[dict] = []
    difficulty: list[dict] = []
    diagnostics: list[dict] = []
    formula = (
        "{score} ~ C(group_code, Treatment(reference='10')) "
        "* C(panel, Treatment(reference='Expert')) "
        "+ C(difficulty, Treatment(reference='Simple'))"
    )
    for dimension, score in OUTCOMES.items():
        model_data = data.dropna(subset=[score]).copy()
        print(f"Fitting {dimension}: n={len(model_data)}", flush=True)
        model = smf.mixedlm(
            formula.format(score=score),
            data=model_data,
            groups=model_data["rater"],
            re_formula="1",
            vc_formula={
                "question": "0 + C(question_id)",
                "response": "0 + C(response_id)",
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = model.fit(reml=False, method="lbfgs", maxiter=2500, disp=False)
                optimizer = "lbfgs"
            except Exception:
                fit = model.fit(reml=False, method="powell", maxiter=2500, disp=False)
                optimizer = "powell"

        for source_code, source_name in SOURCE_NAMES.items():
            source_term = (
                "C(group_code, Treatment(reference='10'))"
                f"[T.{source_code}]"
            )
            interaction_term = (
                "C(group_code, Treatment(reference='10'))"
                f"[T.{source_code}]:C(panel, Treatment(reference='Expert'))[T.Parent]"
            )
            expert = linear_contrast(fit, {source_term: 1.0})
            expert.update(
                dimension=dimension,
                panel="Expert",
                group_code=source_code,
                group_name=source_name,
                comparison=f"{source_name} vs clinic physicians",
            )
            contrasts.append(expert)
            parent = linear_contrast(fit, {source_term: 1.0, interaction_term: 1.0})
            parent.update(
                dimension=dimension,
                panel="Parent",
                group_code=source_code,
                group_name=source_name,
                comparison=f"{source_name} vs clinic physicians",
            )
            contrasts.append(parent)

        difficulty_term = "C(difficulty, Treatment(reference='Simple'))[T.Hard]"
        hard = linear_contrast(fit, {difficulty_term: 1.0})
        hard.update(dimension=dimension, comparison="Hard vs simple")
        difficulty.append(hard)
        diagnostics.append(
            {
                "dimension": dimension,
                "n": int(len(model_data)),
                "converged": bool(fit.converged),
                "optimizer": optimizer,
                "llf": float(fit.llf),
                "aic": float(fit.aic),
                "warnings": [str(item.message) for item in caught],
                "params": {k: float(v) for k, v in fit.params.items()},
            }
        )
        print(f"Finished {dimension}: converged={fit.converged}", flush=True)

    _, adjusted, _, _ = multipletests([row["p_raw"] for row in contrasts], method="holm")
    for row, value in zip(contrasts, adjusted):
        row["p_holm"] = float(value)
    _, adjusted, _, _ = multipletests([row["p_raw"] for row in difficulty], method="holm")
    for row, value in zip(difficulty, adjusted):
        row["p_holm"] = float(value)
    return contrasts, difficulty, diagnostics


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/analysis_dataset.csv")
    parser.add_argument("--out", type=Path, default=root / "results/primary")
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    for column in ("group_code", "question_id", "response_id"):
        data[column] = data[column].astype(str)
    contrasts, difficulty, diagnostics = fit_models(data)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(contrasts).to_csv(args.out / "primary_unadjusted_contrasts.csv", index=False)
    pd.DataFrame(difficulty).to_csv(args.out / "primary_unadjusted_difficulty.csv", index=False)
    (args.out / "primary_unadjusted_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
