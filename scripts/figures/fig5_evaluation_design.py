"""Generate Fig. 5, the evaluation-design schematic."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/main_figures")
args = parser.parse_args()
OUT = args.out_dir
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def create_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 9.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, label, face="white", edge="#222222", size=8.0, weight="normal"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=size, weight=weight, linespacing=1.12)

    def group_frame(x, y, w, h, title, edge, face):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                               linewidth=1.4, linestyle=(0, (5, 3))))
        ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="center",
                fontsize=9.2, weight="bold", color="#222222")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#222222"))

    box(3.55, 8.98, 4.90, 0.78,
        r"Expert-selected evaluation set ($N$ = 100)" "\n50 simple | 50 hard questions",
        face="#F3F3F3", size=8.55, weight="bold")

    group_frame(0.25, 5.7, 3.25, 2.85, "Human responses (4 sources)", "#6B8FC8", "#F4F7FC")
    group_frame(4.35, 5.7, 3.30, 2.85, "FV-LLM configurations (3 sources)", "#63A987", "#F2FAF6")
    group_frame(8.50, 5.7, 3.25, 2.85, "General-purpose / base LLMs (6 sources)", "#CF8C72", "#FFF7F2")

    box(0.55, 7.35, 2.65, 0.55, r"Vaccination-clinic physicians ($n$ = 10)", size=7.2)
    box(0.55, 6.66, 2.65, 0.55, r"General practitioners ($n$ = 10)", size=7.4)
    box(0.55, 5.97, 2.65, 0.55,
        r"Two distinct parent groups ($n$ = 10 each):" "\nwithout internet | with web search",
        size=6.45)
    box(4.67, 7.35, 2.66, 0.55, "Full Model (RAG + Focus + tools)", size=7.35)
    box(4.67, 6.66, 2.66, 0.55, "RAG", size=7.8)
    box(4.67, 5.97, 2.66, 0.55, "RAG + Focus", size=7.8)
    box(8.80, 7.35, 2.65, 0.55, "GPT-4o | Qwen3-Plus", size=7.5)
    box(8.80, 6.66, 2.65, 0.55, "Claude Sonnet 4 | GPT-5", size=7.4)
    box(8.80, 5.97, 2.65, 0.55, "DeepSeek-chat | Gemini 2.5 Pro", size=7.2)

    arrow(6.0, 8.98, 1.88, 8.55)
    arrow(6.0, 8.98, 6.0, 8.55)
    arrow(6.0, 8.98, 10.12, 8.55)

    box(3.80, 4.70, 4.40, 0.68,
        r"Anonymized response pool ($N$ = 1,300)" "\nSource labels and identifying cues removed",
        face="#F7F7F7", size=8.2, weight="bold")
    arrow(1.88, 5.70, 4.65, 5.38)
    arrow(6.0, 5.70, 6.0, 5.38)
    arrow(10.12, 5.70, 7.35, 5.38)

    ax.add_patch(Rectangle((0.35, 0.25), 11.30, 4.00, facecolor="#FCFCFC",
                           edgecolor="#747474", linewidth=1.3, linestyle=(0, (5, 3))))
    ax.text(6.0, 4.00, "Response-source-blinded dual-perspective evaluation",
            ha="center", va="center", fontsize=9.5, weight="bold")
    box(1.05, 2.90, 4.25, 0.72,
        r"Expert panel ($n$ = 10)" "\n1,300 ratings; one intended rating per response",
        face="#EEF5FD", edge="#6B8FC8", size=8.1)
    box(6.70, 2.90, 4.25, 0.72,
        r"Parent panel ($n$ = 16)" "\n1,300 ratings; one intended rating per response",
        face="#FDEFF2", edge="#CC7B8D", size=8.1)
    arrow(5.0, 4.70, 3.18, 3.62)
    arrow(7.0, 4.70, 8.82, 3.62)
    box(0.85, 1.52, 10.30, 0.90,
        "Six 1–5 Likert items\nCorrectness/reliability | Clarity | Completeness\n"
        "Empathy | Fair and non-misleading | Perceived absence of bias",
        size=7.1)
    arrow(3.18, 2.90, 3.18, 2.42)
    arrow(8.82, 2.90, 8.82, 2.42)
    box(1.05, 0.38, 9.90, 0.75,
        "Primary models: source × panel + difficulty\n"
        "Random intercepts: evaluator, question, question-by-source response | Sensitivity: add response length",
        face="#F1FAF6", edge="#63A987", size=7.5)
    arrow(6.0, 1.52, 6.0, 1.13)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
    fig.savefig(OUT / "Fig5.png", dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Fig5.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    create_figure()
