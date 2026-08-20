#!/usr/bin/env python3
"""Run the complete FV-LLM analysis and code-generated figure workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(*arguments: object) -> None:
    command = [sys.executable, *(str(value) for value in arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/analysis_dataset.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    if not data.exists():
        raise FileNotFoundError(
            f"Restricted analysis dataset not found: {data}. See data/README.md."
        )

    run("scripts/data/validate_analysis_dataset.py", "--data", data)
    run("scripts/analysis/run_primary_models.py", "--data", data, "--out", "results/primary")
    run("scripts/analysis/run_length_adjusted_models.py", "--data", data, "--out", "results/length_adjusted")
    run("scripts/analysis/run_pca.py", "--data", data)
    run("scripts/analysis/summarize_response_lengths.py", "--data", data)
    run("scripts/analysis/summarize_standardized_scores.py", "--data", data)
    run("scripts/analysis/run_fig3_difficulty_models.py", "--data", data, "--out", "results/difficulty/fig3_emmeans.csv")
    run("scripts/analysis/run_figs4_difficulty_models.py", "--data", data, "--out", "results/difficulty/figs4_emmeans.csv")

    for script in (
        "fig1_standardized_scores.py",
        "fig2_figs2_radar.py",
        "fig3_difficulty_profiles.py",
        "fig5_evaluation_design.py",
        "figs1_pca.py",
        "figs3_primary_contrasts.py",
        "figs4_difficulty_profiles.py",
    ):
        run(ROOT / "scripts/figures" / script)


if __name__ == "__main__":
    main()
