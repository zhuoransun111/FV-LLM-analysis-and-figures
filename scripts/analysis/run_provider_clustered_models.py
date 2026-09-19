"""Fit sensitivity models with clustering by human response provider.

The human conditions contain 10 anonymous response providers per source, with
10 answers per provider. A crossed variance-component specification is used so
that evaluator, question, response and human-provider intercepts are shared
across the complete dataset rather than being nested within evaluator.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


OUTCOMES = {
    "Correctness/reliability": "Q1OEAE",
    "Clarity": "Q2OEAA",
    "Completeness": "Q3OEAA",
    "Empathy": "Q4OEOEOE",
    "Fair and nonmisleading": "Q5OEAA",
    "Perceived absence of bias": "Q6OEAEOEOE",
}
SOURCE_NAMES = {"2": "RAG", "3": "RAG+Focus"}
HUMAN_SOURCE_CODES = {"10", "11", "12", "13"}


def linear_contrast(fit, weights: dict[str, float]) -> dict[str, float]:
    names = list(fit.fe_params.index)
    vector = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    estimate = float(vector @ fit.fe_params.to_numpy())
    covariance = fit.cov_params().loc[names, names].to_numpy()
    se = float(np.sqrt(vector @ covariance @ vector))
    p_value = float(2 * norm.sf(abs(estimate / se)))
    return {
        "beta": estimate,
        "se": se,
        "ci_low": estimate - 1.96 * se,
        "ci_high": estimate + 1.96 * se,
        "p_raw": p_value,
    }


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "group_code",
        "panel",
        "difficulty",
        "rater",
        "question_id",
        "response_id",
        "response_provider",
        *OUTCOMES.values(),
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prepared = data.copy()
    prepared["group_code"] = prepared["group_code"].astype(str)
    raw_human = prepared[prepared["group_code"].isin(HUMAN_SOURCE_CODES)]
    if raw_human["response_provider"].isna().any():
        raise ValueError("Human-source rows require anonymous response_provider values.")
    for column in ("question_id", "response_id", "response_provider"):
        prepared[column] = prepared[column].astype(str)
    prepared["response_provider"] = prepared["response_provider"].str.strip()
    prepared["human_source"] = prepared["group_code"].isin(HUMAN_SOURCE_CODES).astype(float)
    human = prepared[prepared["human_source"].eq(1)]
    if human["response_provider"].isin({"", "nan", "none", "null"}).any():
        raise ValueError("Human-source rows require anonymous response_provider values.")
    providers_per_source = human.groupby("group_code")["response_provider"].nunique()
    if not providers_per_source.eq(10).all():
        raise ValueError(
            "Expected 10 anonymous response providers per human source; "
            f"found {providers_per_source.to_dict()}."
        )
    provider_source_counts = human.groupby("response_provider")["group_code"].nunique()
    if not provider_source_counts.eq(1).all() or human["response_provider"].nunique() != 40:
        raise ValueError(
            "Expected 40 source-specific anonymous response providers, with no "
            "provider code reused across human sources."
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
            "one expert-panel and one parent-panel rating record."
        )

    # Nonhuman rows receive an existing placeholder category multiplied by 0.
    # This keeps their provider design rows exactly zero without adding a
    # nonhuman pseudo-provider random effect.
    placeholder = human["response_provider"].iloc[0]
    prepared["provider_vc"] = prepared["response_provider"].where(
        prepared["human_source"].eq(1), placeholder
    )
    prepared["all_observations"] = "all"
    return prepared


def fit_models(data: pd.DataFrame, adjust_length: bool = False):
    data = prepare_data(data)
    contrasts: list[dict] = []
    difficulty: list[dict] = []
    length: list[dict] = []
    diagnostics: list[dict] = []
    formula = (
        "{score} ~ C(group_code, Treatment(reference='10')) "
        "* C(panel, Treatment(reference='Expert')) "
        "+ C(difficulty, Treatment(reference='Simple'))"
    )
    if adjust_length:
        if "length_100" not in data.columns:
            raise ValueError("length_100 is required for the length-adjusted model.")
        formula += " + length_100"

    vc_formula = {
        "evaluator": "0 + C(rater)",
        "question": "0 + C(question_id)",
        "response": "0 + C(response_id)",
        "human_provider": "0 + C(provider_vc):human_source",
    }

    for dimension, score in OUTCOMES.items():
        model_data = data.dropna(subset=[score]).copy()
        print(f"Fitting {dimension}: n={len(model_data)}", flush=True)
        model = smf.mixedlm(
            formula.format(score=score),
            data=model_data,
            groups=model_data["all_observations"],
            re_formula="0",
            vc_formula=vc_formula,
            use_sparse=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = None
            failures: list[str] = []
            for optimizer in ("lbfgs", "powell", "cg"):
                try:
                    candidate = model.fit(
                        reml=False, method=optimizer, maxiter=2500, disp=False
                    )
                except Exception as exc:
                    failures.append(f"{optimizer}: {type(exc).__name__}: {exc}")
                    continue
                fit = candidate
                if candidate.converged:
                    break
                failures.append(f"{optimizer}: did not converge")
            if fit is None:
                raise RuntimeError("All optimizers failed: " + "; ".join(failures))

        for source_code, source_name in SOURCE_NAMES.items():
            source_term = (
                "C(group_code, Treatment(reference='10'))" f"[T.{source_code}]"
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
            parent = linear_contrast(
                fit, {source_term: 1.0, interaction_term: 1.0}
            )
            parent.update(
                dimension=dimension,
                panel="Parent",
                group_code=source_code,
                group_name=source_name,
                comparison=f"{source_name} vs clinic physicians",
            )
            contrasts.append(parent)

        difficulty_row = linear_contrast(
            fit, {"C(difficulty, Treatment(reference='Simple'))[T.Hard]": 1.0}
        )
        difficulty_row.update(dimension=dimension, comparison="Hard vs simple")
        difficulty.append(difficulty_row)

        if adjust_length:
            length_row = linear_contrast(fit, {"length_100": 1.0})
            length_row.update(
                dimension=dimension,
                comparison="Per 100 additional cleaned Chinese characters",
            )
            length.append(length_row)

        component_variances = {
            name: float(value)
            for name, value in zip(model.exog_vc.names, fit.vcomp)
        }
        diagnostics.append(
            {
                "dimension": dimension,
                "formula": formula.format(score=score),
                "n": int(len(model_data)),
                "converged": bool(fit.converged),
                "optimizer": optimizer,
                "llf": float(fit.llf),
                "aic": float(fit.aic),
                "variance_components": component_variances,
                "warnings": [str(item.message) for item in caught],
                "optimizer_attempts": failures,
            }
        )
        print(
            f"Finished {dimension}: converged={fit.converged}; "
            f"human-provider variance={component_variances['human_provider']:.6f}",
            flush=True,
        )

    for family in (contrasts, difficulty, length):
        if not family:
            continue
        _, adjusted, _, _ = multipletests(
            [row["p_raw"] for row in family], method="holm"
        )
        for row, value in zip(family, adjusted):
            row["p_holm"] = float(value)
    return contrasts, difficulty, length, diagnostics


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/analysis_dataset.csv")
    parser.add_argument(
        "--out", type=Path, default=root / "results/provider_clustered"
    )
    parser.add_argument("--adjust-length", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    contrasts, difficulty, length, diagnostics = fit_models(
        data, adjust_length=args.adjust_length
    )
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = "length_adjusted" if args.adjust_length else "primary"
    pd.DataFrame(contrasts).to_csv(
        args.out / f"{prefix}_provider_clustered_contrasts.csv", index=False
    )
    pd.DataFrame(difficulty).to_csv(
        args.out / f"{prefix}_provider_clustered_difficulty.csv", index=False
    )
    if length:
        pd.DataFrame(length).to_csv(
            args.out / "length_adjusted_provider_clustered_length_effects.csv",
            index=False,
        )
    (args.out / f"{prefix}_provider_clustered_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
