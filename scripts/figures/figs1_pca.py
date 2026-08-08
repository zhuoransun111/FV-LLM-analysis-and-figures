"""Generate Supplementary Fig. S1 from panel-specific PCA loadings."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Figure style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 600
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

labels = [
    'Correctness/reliability', 'Clarity', 'Completeness', 'Empathy',
    'Fair/non-misleading', 'No perceived bias'
]

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input", type=Path, default=ROOT / "results/descriptive/pca_loadings.csv"
)
parser.add_argument(
    "--summary", type=Path, default=ROOT / "results/descriptive/pca_summary.json"
)
parser.add_argument(
    "--out-dir", type=Path, default=ROOT / "outputs/supplementary_figures"
)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

loadings = pd.read_csv(args.input)
summaries = {row["panel"]: row for row in json.loads(args.summary.read_text())}
expert = loadings.loc[loadings["panel"].eq("Expert")]
parent = loadings.loc[loadings["panel"].eq("Parent")]
expert_pc1 = expert["PC1_loading"].to_numpy()
expert_pc2 = expert["PC2_loading"].to_numpy()
parent_pc1 = parent["PC1_loading"].to_numpy()
parent_pc2 = parent["PC2_loading"].to_numpy()

# Nature double-column width is approximately 180 mm (7.08 inches).
fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.8))
print("Fig. S1: canvas created", flush=True)

# Okabe-Ito colour-blind-safe palette.
colors = ['#D55E00', '#0072B2', '#009E73', '#F0E442', '#CC79A7', '#56B4E9']
markers = ['o', 's', '^', 'D', 'v', 'p']


def format_axes(ax, title, panel_letter):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=3, width=0.8, direction='out')

    ax.set_title(title, fontsize=9, fontweight='bold', pad=8)

    ax.text(-0.15, 1.05, panel_letter, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='right')


# Panel a: expert evaluators
ax1 = axes[0]
for i in range(len(labels)):
    ax1.scatter(expert_pc1[i], expert_pc2[i], c=colors[i], marker=markers[i], s=60,
                edgecolors='black', linewidths=0.5, label=labels[i], zorder=3)

ax1.axhline(0, color='gray', linestyle='--', linewidth=0.6, zorder=1)
ax1.axvline(np.mean(expert_pc1), color='gray', linestyle='--', linewidth=0.6, zorder=1)

ax1.set_xlabel(f"PC1 ({summaries['Expert']['PC1_explained_variance_percent']:.2f}% variance explained)")
ax1.set_ylabel(f"PC2 ({summaries['Expert']['PC2_explained_variance_percent']:.2f}% variance explained)")
format_axes(ax1, 'Expert population', 'a')

# Panel b: parent evaluators
ax2 = axes[1]
for i in range(len(labels)):
    ax2.scatter(parent_pc1[i], parent_pc2[i], c=colors[i], marker=markers[i], s=60,
                edgecolors='black', linewidths=0.5, label=labels[i], zorder=3)

ax2.axhline(0, color='gray', linestyle='--', linewidth=0.6, zorder=1)
ax2.axvline(np.mean(parent_pc1), color='gray', linestyle='--', linewidth=0.6, zorder=1)

ax2.set_xlabel(f"PC1 ({summaries['Parent']['PC1_explained_variance_percent']:.2f}% variance explained)")
ax2.set_ylabel(f"PC2 ({summaries['Parent']['PC2_explained_variance_percent']:.2f}% variance explained)")
format_axes(ax2, 'Parent population', 'b')

# Global legend
handles, lbls = ax1.get_legend_handles_labels()
fig.legend(handles, lbls, loc='upper center', bbox_to_anchor=(0.5, 0.985),
           ncol=6, frameon=False, handletextpad=0.3, columnspacing=1.2)

# Keep the legend inside the canvas. This avoids the very slow tight-bounding-
# box calculation observed for an outside legend at 600 dpi.
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.78, wspace=0.34)

print("Fig. S1: saving PDF", flush=True)
plt.savefig(args.out_dir / 'FigS1.pdf', format='pdf', facecolor='white')
print("Fig. S1: saving PNG", flush=True)
plt.savefig(args.out_dir / 'FigS1.png', dpi=600, facecolor='white')
print("Fig. S1: files saved", flush=True)
print(args.out_dir / 'FigS1.png')
print(args.out_dir / 'FigS1.pdf')
