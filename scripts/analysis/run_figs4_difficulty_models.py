"""Estimate adjusted difficulty profiles used in Supplementary Fig. S4."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd
import statsmodels.formula.api as smf


OUTCOMES = {
    "Accuracy": "Q1OEAE",
    "Clarity": "Q2OEAA",
    "Completeness": "Q3OEAA",
    "Empathy": "Q4OEOEOE",
    "Safety": "Q5OEAA",
    "Perceived absence of bias": "Q6OEAEOEOE",
}

GROUP_NAMES = {
    "1": "FV-LLM full configuration",
    "2": "RAG",
    "3": "RAG+Focus",
    "4": "GPT-4o",
    "5": "Qwen3-Plus",
    "6": "Claude Sonnet 4",
    "7": "GPT-5",
    "8": "DeepSeek-chat",
    "9": "Gemini 2.5 Pro",
    "10": "Vaccination-clinic physicians",
    "11": "Parents without internet",
    "12": "Parents with web search",
    "13": "General practitioners",
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/analysis_dataset.csv")
    parser.add_argument("--out", type=Path, default=root / "results/difficulty/figs4_emmeans.csv")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data["group_code"] = data["group_code"].astype(str)
    data["question_id"] = data["question_id"].astype(str)
    data["response_id"] = data["response_id"].astype(str)
    mean_length = float(data["length_100"].mean())

    formula = (
        "{score} ~ C(group_code, Treatment(reference='10')) "
        "* C(panel, Treatment(reference='Expert')) "
        "* C(difficulty, Treatment(reference='Simple')) + length_100"
    )

    predictions: list[dict[str, object]] = []
    for outcome, score in OUTCOMES.items():
        model_data = data.dropna(subset=[score]).copy()
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = model.fit(reml=False, method="lbfgs", maxiter=2500, disp=False)
            except Exception:
                fit = model.fit(reml=False, method="powell", maxiter=2500, disp=False)

        grid = pd.DataFrame(
            [
                {
                    "group_code": group,
                    "panel": panel,
                    "difficulty": difficulty,
                    "length_100": mean_length,
                }
                for group in GROUP_NAMES
                for panel in ("Expert", "Parent")
                for difficulty in ("Simple", "Hard")
            ]
        )
        grid["adjusted_mean"] = fit.predict(grid)
        for row in grid.to_dict("records"):
            predictions.append(
                {
                    "outcome": outcome,
                    "panel": row["panel"],
                    "group_code": row["group_code"],
                    "source": GROUP_NAMES[row["group_code"]],
                    "difficulty": row["difficulty"],
                    "adjusted_mean": float(row["adjusted_mean"]),
                    "mean_length_100": mean_length,
                    "converged": bool(fit.converged),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
