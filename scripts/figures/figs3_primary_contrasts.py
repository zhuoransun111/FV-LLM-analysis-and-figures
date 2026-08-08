"""Generate high-resolution Supplementary Fig. S3 from archived primary contrasts."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input",
    type=Path,
    default=ROOT / "results/primary/primary_unadjusted_contrasts.csv",
)
parser.add_argument(
    "--out-dir", type=Path, default=ROOT / "outputs/supplementary_figures"
)
args = parser.parse_args()
DATA = args.input
OUT = args.out_dir
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

dimensions = [
    "Correctness/reliability",
    "Clarity",
    "Completeness",
    "Empathy",
    "Fairness/non-misleadingness",
    "Perceived absence of bias",
]
labels = [
    "Correctness/\nreliability",
    "Clarity",
    "Completeness",
    "Empathy",
    "Fairness/non-\nmisleadingness",
    "Perceived absence\nof bias",
]
rows = pd.read_csv(DATA)
lookup = {(row.dimension, row.panel, row.group_name): row for row in rows.itertuples(index=False)}
y = list(range(len(dimensions)))[::-1]
colors = {"RAG": "#1565C0", "RAG+Focus": "#D97706"}
offsets = {"RAG": 0.13, "RAG+Focus": -0.13}

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), sharey=True)
for ax, panel, letter in zip(axes, ("Expert", "Parent"), ("a", "b")):
    for source in ("RAG", "RAG+Focus"):
        selected = [lookup[(dimension, panel, source)] for dimension in dimensions]
        beta = [float(row.beta) for row in selected]
        low = [float(row.ci_low) for row in selected]
        high = [float(row.ci_high) for row in selected]
        yy = [value + offsets[source] for value in y]
        ax.errorbar(
            beta,
            yy,
            xerr=[
                [b - l for b, l in zip(beta, low)],
                [h - b for b, h in zip(beta, high)],
            ],
            fmt="o",
            markersize=5.4,
            linewidth=1.15,
            color=colors[source],
            ecolor=colors[source],
            capsize=2.8,
            label=source,
        )
    ax.axvline(0, color="#777777", linewidth=0.9, linestyle="--")
    ax.set_title(f"{panel} panel", fontsize=10, fontweight="bold")
    ax.set_xlabel("Difference from vaccination-clinic physicians (β)", fontsize=8.2)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.65)
    ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

for ax in axes:
    ax.set_yticks(y, labels, fontsize=8)
    ax.tick_params(labelleft=True)
axes[1].legend(frameon=False, loc="lower right", fontsize=8.5)
fig.tight_layout()

png = OUT / "FigS3.png"
pdf = OUT / "FigS3.pdf"
fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(png)
print(pdf)
