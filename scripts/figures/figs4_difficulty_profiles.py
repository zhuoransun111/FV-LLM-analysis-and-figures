"""Generate the final high-resolution Supplementary Fig. S4.

Changes from the earlier long figure:
- Times New Roman throughout;
- no stars and no promotional group names;
- neutral source-group labels in a compact top legend;
- group membership indicated by light horizontal bands instead of unreadable
  vertical labels;
- ``Delta simple-hard`` replaces ``Drop``;
- a 9 x 13.5 inch canvas designed for full-page supplementary placement;
- 600 dpi PNG plus vector PDF.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input", type=Path, default=ROOT / "results/difficulty/figs4_emmeans.csv"
)
parser.add_argument(
    "--out-dir", type=Path, default=ROOT / "outputs/supplementary_figures"
)
args = parser.parse_args()
EMM_FILE = args.input
OUT_DIR = args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.linewidth": 0.75,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

emm = pd.read_csv(EMM_FILE)

GROUPS = {
    "FV-LLM configurations": ["5", "1", "2", "3"],
    "General-purpose LLMs": ["9", "8", "6", "4", "7"],
    "Human responses": ["11", "12", "13", "10"],
}
ORDER = [code for codes in GROUPS.values() for code in codes]
LABELS = {
    "1": "Full configuration",
    "2": "RAG",
    "3": "RAG+Focus",
    "4": "GPT-4o",
    "5": "Qwen3-Plus",
    "6": "Claude Sonnet 4",
    "7": "GPT-5",
    "8": "DeepSeek-chat",
    "9": "Gemini 2.5 Pro",
    "10": "Vaccination-clinic physicians",
    "11": "Parents (no internet)",
    "12": "Parents (web search)",
    "13": "General practitioners",
}
SOURCE_TO_CODE = {
    "FV-LLM full configuration": "1",
    "RAG": "2",
    "RAG+Focus": "3",
    "GPT-4o": "4",
    "Qwen3-Plus": "5",
    "Claude Sonnet 4": "6",
    "GPT-5": "7",
    "DeepSeek-chat": "8",
    "Gemini 2.5 Pro": "9",
    "Vaccination-clinic physicians": "10",
    "Parents without internet": "11",
    "Parents with web search": "12",
    "General practitioners": "13",
}
OUTCOME_LABELS = {
    "Accuracy": "Correctness/reliability",
    "Safety": "Fair and non-misleading",
    "Completeness": "Completeness",
    "Clarity": "Clarity",
    "Empathy": "Empathy",
    "Perceived absence of bias": "Perceived absence of bias",
}
PANELS = [
    ("Accuracy", "a", "b"),
    ("Safety", "c", "d"),
    ("Completeness", "e", "f"),
    ("Clarity", "g", "h"),
    ("Empathy", "i", "j"),
    ("Perceived absence of bias", "k", "l"),
]

COLORS = {
    "FV-LLM configurations": ("#B9DDF5", "#2B6CB0", "#EDF5FB"),
    "General-purpose LLMs": ("#F8BDC7", "#C94C67", "#FDF0F3"),
    "Human responses": ("#CED5DB", "#56616B", "#F1F3F5"),
}


def group_for(code: str) -> str:
    for group, codes in GROUPS.items():
        if code in codes:
            return group
    raise KeyError(code)


def values(panel: str, outcome: str) -> dict[str, tuple[float, float]]:
    subset = emm.loc[(emm["panel"] == panel) & (emm["outcome"] == outcome)]
    result: dict[str, dict[str, float]] = {}
    for row in subset.itertuples(index=False):
        code = SOURCE_TO_CODE[row.source]
        result.setdefault(code, {})[row.difficulty] = float(row.adjusted_mean)
    return {code: (result[code]["Simple"], result[code]["Hard"]) for code in ORDER}


def draw_panel(ax, panel: str, outcome: str, letter: str, show_xlabel: bool) -> None:
    panel_values = values(panel, outcome)
    y = np.arange(len(ORDER))[::-1]

    # Three subtle bands replace the previous vertical promotional labels.
    bounds = [(8.5, 12.5), (3.5, 8.5), (-0.5, 3.5)]
    for (group, _), (low, high) in zip(GROUPS.items(), bounds):
        ax.axhspan(low, high, color=COLORS[group][2], zorder=0)

    for idx, code in enumerate(ORDER):
        yy = y[idx]
        simple, hard = panel_values[code]
        group = group_for(code)
        light, dark, _ = COLORS[group]
        ax.plot([simple, hard], [yy, yy], color="#C8CDD2", linewidth=1.35, zorder=2)
        ax.scatter(simple, yy, s=30, color=light, edgecolor="white", linewidth=0.55, zorder=4)
        ax.scatter(hard, yy, s=30, color=dark, edgecolor="white", linewidth=0.55, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[code] for code in ORDER], fontsize=7.3)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", labelsize=7, length=2.5, width=0.7)
    ax.set_xlim(1.45, 5.05)
    ax.set_xticks([2, 3, 4, 5])
    ax.grid(axis="x", color="#E1E4E8", linewidth=0.55, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(
        f"{panel}: {OUTCOME_LABELS[outcome]}",
        fontsize=8.1,
        fontweight="bold",
        pad=5,
    )
    ax.text(
        -0.31,
        1.03,
        letter,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    if show_xlabel:
        ax.set_xlabel(
            "Adjusted rating (estimated marginal mean on the 1–5 scale)",
            fontsize=7.1,
            labelpad=3,
        )
    else:
        ax.tick_params(axis="x", labelbottom=False)


legend_items = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#B9DDF5", markeredgecolor="white", markersize=5.5, label="Simple questions"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#2B6CB0", markeredgecolor="white", markersize=5.5, label="Hard questions"),
    Patch(facecolor=COLORS["FV-LLM configurations"][2], edgecolor="none", label="FV-LLM configurations"),
    Patch(facecolor=COLORS["General-purpose LLMs"][2], edgecolor="none", label="General-purpose LLMs"),
    Patch(facecolor=COLORS["Human responses"][2], edgecolor="none", label="Human responses"),
]

def render(panel_specs, figsize, stem: str, continued: bool = False) -> None:
    fig, axes = plt.subplots(len(panel_specs), 2, figsize=figsize, squeeze=False)
    for row, (outcome, left_letter, right_letter) in enumerate(panel_specs):
        show_xlabel = row == len(panel_specs) - 1
        draw_panel(axes[row, 0], "Expert", outcome, left_letter, show_xlabel)
        draw_panel(axes[row, 1], "Parent", outcome, right_letter, show_xlabel)
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.997),
        ncol=5,
        frameon=False,
        fontsize=7.1,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    if continued:
        fig.text(0.012, 0.992, "Fig. S4 (continued)", ha="left", va="top", fontsize=7.5)
    fig.subplots_adjust(
        left=0.245,
        right=0.985,
        top=0.905 if len(panel_specs) == 3 else 0.958,
        bottom=0.075 if len(panel_specs) == 3 else 0.055,
        wspace=0.46,
        hspace=0.48,
    )
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


# A complete source file is retained for submission systems that require one
# image, while the two half-height renderings are intended for the Word
# supplementary document so text remains readable at page width.
render(PANELS, (9.0, 13.5), "FigS4")
render(PANELS[:3], (9.0, 7.5), "FigS4_part1_panels_a-f")
render(PANELS[3:], (9.0, 7.5), "FigS4_part2_panels_g-l", continued=True)
