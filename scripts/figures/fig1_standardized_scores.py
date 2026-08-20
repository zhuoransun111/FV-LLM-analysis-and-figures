"""Generate Fig. 1 from aggregate within-rater standardized scores."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Figure style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 600
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

COLOR_PALETTE = {
    'FullModel': ['#A2D2FF', '#BDE0FE', '#4895EF', '#4361EE'],
    'General': ['#FFC2D1', '#FB6F92', '#FF85A1', '#FFE5EC', '#FFB3C1'],
    'Human': ['#E9ECEF', '#CED4DA', '#ADB5BD', '#6C757D'],
}

COMPARISON_HIGHLIGHT = '#4361EE'

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input",
    type=Path,
    default=ROOT / "results/descriptive/standardized_scores.csv",
)
parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/main_figures")
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

# Rating-item labels used in the aggregate input file.
dimensions = ['Q1_Accuracy', 'Q2_Clarity', 'Q3_Completeness', 'Q4_Empathy', 'Q5_Safety', 'Q6_Bias']
dim_labels = [
    'Correctness/\nreliability', 'Clarity', 'Completeness', 'Empathy',
    'Fair and non-\nmisleading', 'Perceived absence\nof bias'
]


label_to_dimension = {
    'Correctness/reliability': 'Q1_Accuracy',
    'Clarity': 'Q2_Clarity',
    'Completeness': 'Q3_Completeness',
    'Empathy': 'Q4_Empathy',
    'Fair and non-misleading': 'Q5_Safety',
    'Perceived absence of bias': 'Q6_Bias',
}


def load_summary(panel):
    summary = pd.read_csv(args.input)
    summary = summary.loc[summary['panel'].eq(panel)].copy()
    summary['GROUP'] = summary['group_code'].astype(int)
    summary['dimension'] = summary['rating_item'].map(label_to_dimension)
    if summary['dimension'].isna().any():
        raise ValueError('Unknown rating item in standardized-score summary')
    columns = {}
    for dimension in dimensions:
        subset = summary.loc[summary['dimension'].eq(dimension)].set_index('GROUP')
        columns[(f'{dimension}_Z', 'mean')] = subset['mean_z']
        columns[(f'{dimension}_Z', 'sem')] = subset['sem_z']
    result = pd.DataFrame(columns)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result.sort_index()


expert_agg = load_summary('Expert')
parent_agg = load_summary('Parent')

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(18, 13))

plt.subplots_adjust(hspace=0.65, wspace=0.2, top=0.90, bottom=0.1)

fig.text(0.28, 0.94, 'Expert evaluation', fontsize=15, fontweight='bold', ha='center')
fig.text(0.72, 0.94, 'Parent evaluation', fontsize=15, fontweight='bold', ha='center')


def plot_refined_subplot(ax, agg_df, group_ids, group_names, colors, panel_letter):
    x = np.arange(len(dimensions))
    width = 0.85 / len(group_ids)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#F0F0F0', linestyle='-', linewidth=0.7, zorder=0)
    ax.axhline(0, color='#444444', linewidth=0.8, zorder=1)

    for i, (g_id, g_name, color) in enumerate(zip(group_ids, group_names, colors)):
        means = [agg_df.loc[g_id, (f'{dim}_Z', 'mean')] for dim in dimensions]
        sems = [agg_df.loc[g_id, (f'{dim}_Z', 'sem')] for dim in dimensions]

        pos = x + (i - len(group_ids) / 2 + 0.5) * width
        ax.bar(pos, means, width, yerr=sems, label=g_name,
               color=color, alpha=0.9, edgecolor='white', linewidth=0.5, zorder=3,
               error_kw={'elinewidth': 0.8, 'capsize': 1.2, 'ecolor': '#333333'})

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=9, fontweight='bold')
    ax.set_ylabel(r'Score ($Z$-score)', fontsize=9)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=len(group_ids),
              frameon=False, fontsize=9, handletextpad=0.2, columnspacing=1.2)

    ax.text(-0.06, 1.15, panel_letter, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top')


# Source groups shown in the expert-panel column.
configs_expert = [
    {'ids': [5, 1, 2, 3], 'names': ['Qwen3-Plus', 'Full Model', 'RAG', 'RAG+Focus'],
     'colors': COLOR_PALETTE['FullModel']},
    {'ids': [4, 6, 7, 8, 9, 2], 'names': ['GPT-4o', 'Claude Sonnet 4', 'GPT-5', 'DeepSeek-chat', 'Gemini 2.5 Pro', 'RAG'],
     'colors': COLOR_PALETTE['General'] + [COMPARISON_HIGHLIGHT]},
    {'ids': [10, 11, 12, 13, 2], 'names': ['Clinic physicians', 'Parents: no internet', 'Parents: web search', 'General practitioners', 'RAG'],
     'colors': COLOR_PALETTE['Human'] + [COMPARISON_HIGHLIGHT]}
]

# Source groups shown in the parent-panel column.
configs_parent = [
    {'ids': [5, 1, 2, 3], 'names': ['Qwen3-Plus', 'Full Model', 'RAG', 'RAG+Focus'],
     'colors': COLOR_PALETTE['FullModel']},
    {'ids': [4, 6, 7, 8, 9, 3], 'names': ['GPT-4o', 'Claude Sonnet 4', 'GPT-5', 'DeepSeek-chat', 'Gemini 2.5 Pro', 'RAG+Focus'],
     'colors': COLOR_PALETTE['General'] + [COMPARISON_HIGHLIGHT]},
    {'ids': [10, 11, 12, 13, 3], 'names': ['Clinic physicians', 'Parents: no internet', 'Parents: web search', 'General practitioners', 'RAG+Focus'],
     'colors': COLOR_PALETTE['Human'] + [COMPARISON_HIGHLIGHT]}
]

for row_idx in range(3):
    plot_refined_subplot(axes[row_idx, 0], expert_agg, configs_expert[row_idx]['ids'],
                         configs_expert[row_idx]['names'], configs_expert[row_idx]['colors'], chr(97 + row_idx * 2))

    plot_refined_subplot(axes[row_idx, 1], parent_agg, configs_parent[row_idx]['ids'],
                         configs_parent[row_idx]['names'], configs_parent[row_idx]['colors'], chr(98 + row_idx * 2))

output_png = args.out_dir / "Fig1.png"
output_pdf = args.out_dir / "Fig1.pdf"
plt.savefig(output_png, dpi=600, bbox_inches='tight')
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
print(output_png)
print(output_pdf)
