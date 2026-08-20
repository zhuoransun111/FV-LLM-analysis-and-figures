"""Regenerate Fig. 2 using the author's stated Z-to-1–5 mapping.

The data transformation is unchanged from the supplied plotting script:
mapped = 3 + clip(2 * Z, -2, 2). Only construct-aligned axis wording and
neutral source labels are used here.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input",
    type=Path,
    default=ROOT / "results/descriptive/standardized_scores.csv",
)
parser.add_argument("--main-out", type=Path, default=ROOT / "outputs/main_figures")
parser.add_argument(
    "--supp-out", type=Path, default=ROOT / "outputs/supplementary_figures"
)
args = parser.parse_args()
MAIN_OUT_DIR = args.main_out
SUPP_OUT_DIR = args.supp_out
MAIN_OUT_DIR.mkdir(parents=True, exist_ok=True)
SUPP_OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "figure.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COL_MAP = {
    "group": "GROUP",
    "GROUP": "GROUP",
    "Q1OEAE": "Q1_Accuracy",
    "Q2OEAA": "Q2_Clarity",
    "Q3OEAA": "Q3_Completeness",
    "Q4OEOEOE": "Q4_Empathy",
    "Q5OEAA": "Q5_Safety",
    "Q6OEAEOEOE": "Q6_Bias",
}
DIMS = [
    "Q1_Accuracy",
    "Q2_Clarity",
    "Q3_Completeness",
    "Q4_Empathy",
    "Q5_Safety",
    "Q6_Bias",
]
Z_COLS = [f"{dim}_Z" for dim in DIMS]
LABELS = [
    "Correctness/\nreliability",
    "Clarity",
    "Completeness",
    "Empathy",
    "Fair and non-\nmisleading",
    "Perceived absence\nof bias",
]


def load_panel(panel: str) -> pd.DataFrame:
    data = pd.read_csv(args.input)
    data = data.loc[data["panel"] == panel].copy()
    label_to_dimension = {
        "Correctness/reliability": "Q1_Accuracy_Z",
        "Clarity": "Q2_Clarity_Z",
        "Completeness": "Q3_Completeness_Z",
        "Empathy": "Q4_Empathy_Z",
        "Fair and non-misleading": "Q5_Safety_Z",
        "Perceived absence of bias": "Q6_Bias_Z",
    }
    data["dimension"] = data["rating_item"].map(label_to_dimension)
    result = data.pivot(index="group_code", columns="dimension", values="mean_z")
    result.index = result.index.astype(int)
    return result[Z_COLS].sort_index()


def mapped(panel: pd.DataFrame, group_id: int) -> list[float]:
    z = panel.loc[group_id, Z_COLS].to_numpy(dtype=float)
    values = 3 + np.clip(2 * z, -2, 2)
    return [*values, values[0]]


expert = load_panel("Expert")
parent = load_panel("Parent")

angles = np.linspace(0, 2 * np.pi, len(LABELS), endpoint=False).tolist()
angles += angles[:1]

COLORS = {
    "Clinic physicians": "#E66101",
    "RAG": "#0571B0",
    "GPT-5": "#5E3C99",
    "RAG+Focus": "#92C5DE",
}

fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.54), subplot_kw={"polar": True})


def draw(ax, series: dict[str, list[float]], title: str, letter: str) -> None:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(color="#AAAAAA", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["polar"].set_color("#222222")
    ax.spines["polar"].set_linewidth(1.0)
    ax.fill(angles, [2] * 7, color="#444444", alpha=0.15, zorder=0)
    ax.fill(angles, [4] * 7, color="#444444", alpha=0.08, zorder=0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABELS, fontsize=7.2, fontweight="bold")
    ax.tick_params(axis="x", pad=8)
    ax.set_rlabel_position(0)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color="grey", size=6)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=15)
    ax.text(
        -0.15,
        1.1,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
    )
    for idx, (name, values) in enumerate(series.items(), start=1):
        ax.plot(angles, values, color=COLORS[name], linewidth=1.5, zorder=idx + 1)
        ax.fill(angles, values, color=COLORS[name], alpha=0.13, zorder=1)


draw(
    axes[0],
    {
        "Clinic physicians": mapped(expert, 10),
        "GPT-5": mapped(expert, 7),
        "RAG": mapped(expert, 2),
    },
    "Expert evaluation",
    "a",
)
draw(
    axes[1],
    {
        "Clinic physicians": mapped(parent, 10),
        "GPT-5": mapped(parent, 7),
        "RAG+Focus": mapped(parent, 3),
    },
    "Parent evaluation",
    "b",
)

legend = [
    Line2D([0], [0], color=COLORS[name], linewidth=1.5, label=name)
    for name in ("Clinic physicians", "RAG", "GPT-5", "RAG+Focus")
]
fig.legend(
    handles=legend,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.13),
    ncol=4,
    fontsize=7.5,
    frameon=False,
    columnspacing=1.2,
)
plt.tight_layout()
fig.savefig(MAIN_OUT_DIR / "Fig2.png", dpi=600, bbox_inches="tight")
fig.savefig(MAIN_OUT_DIR / "Fig2.pdf", format="pdf", bbox_inches="tight")
print(MAIN_OUT_DIR / "Fig2.png")


# Supplementary Fig. S2 uses a single global min–max transformation across
# both panels, all response sources and all six dimensions. Regenerate it with
# neutral source names while leaving the transformation unchanged.
plt.close(fig)
all_values = np.concatenate([expert.to_numpy().ravel(), parent.to_numpy().ravel()])
global_min = np.nanmin(all_values)
global_max = np.nanmax(all_values)


def minmax_mapped(panel: pd.DataFrame, group_id: int) -> list[float]:
    z = panel.loc[group_id, Z_COLS].to_numpy(dtype=float)
    values = 1 + 4 * (z - global_min) / (global_max - global_min)
    return [*values, values[0]]


source_order = [10, 13, 12, 11, 7, 4, 6, 9, 8, 5, 1, 2, 3]
source_titles = {
    10: "Vaccination-clinic physicians",
    13: "General practitioners",
    12: "Parent respondents\n(internet access)",
    11: "Parent respondents\n(no internet access)",
    7: "GPT-5",
    4: "GPT-4o",
    6: "Claude Sonnet 4",
    9: "Gemini 2.5 Pro",
    8: "DeepSeek-chat",
    5: "Qwen3-Plus",
    1: "FV-LLM full configuration",
    2: "RAG",
    3: "RAG+Focus",
}

fig_s2, axes_s2 = plt.subplots(4, 4, figsize=(20, 22), subplot_kw={"polar": True})
plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.55, hspace=0.62)
axes_s2 = axes_s2.flatten()
for index, ax in enumerate(axes_s2):
    if index >= len(source_order):
        ax.axis("off")
        continue
    group_id = source_order[index]
    exp_values = minmax_mapped(expert, group_id)
    par_values = minmax_mapped(parent, group_id)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABELS, fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", pad=13)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color="#888888", size=8)
    ax.plot(angles, exp_values, linewidth=2.2, color="#4361EE", zorder=3)
    ax.fill(angles, exp_values, color="#4361EE", alpha=0.22, zorder=2)
    ax.plot(angles, par_values, linewidth=2.2, color="#FB6F92", zorder=3)
    ax.fill(angles, par_values, color="#FB6F92", alpha=0.20, zorder=2)
    ax.set_title(source_titles[group_id], fontsize=12, fontweight="bold", y=1.23)

legend_s2 = [
    Line2D([0], [0], color="#4361EE", lw=3, label="Expert panel"),
    Line2D([0], [0], color="#FB6F92", lw=3, label="Parent panel"),
]
fig_s2.legend(
    handles=legend_s2,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=2,
    frameon=False,
    fontsize=14,
)
fig_s2.text(
    0.5,
    0.945,
    "Within-rater standardised means converted to a 1–5 display scale",
    ha="center",
    fontsize=12,
)
fig_s2.savefig(SUPP_OUT_DIR / "FigS2.png", dpi=600, bbox_inches="tight")
fig_s2.savefig(SUPP_OUT_DIR / "FigS2.pdf", format="pdf", bbox_inches="tight")
print(SUPP_OUT_DIR / "FigS2.png")
