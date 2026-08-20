"""Generate Fig. 3 from adjusted simple- and hard-question means."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 600
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# The visual design follows the original script, while values come from the
# revised source-by-panel-by-difficulty mixed models used in the manuscript.
ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input", type=Path, default=ROOT / "results/difficulty/fig3_emmeans.csv"
)
parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/main_figures")
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
emm_long = pd.read_csv(args.input)

tier_human = ['10']
tier_general = ['4', '7']
tier_fvllm = ['1', '2', '3']

plot_models = tier_human[::-1] + tier_general[::-1] + tier_fvllm[::-1]

model_names_map = {
    '1': 'Full configuration', '2': 'RAG', '3': 'RAG+Focus',
    '4': 'GPT-4o', '7': 'GPT-5', '10': 'Clinic physicians'
}

source_to_model = {
    'FV-LLM full configuration': '1',
    'RAG': '2',
    'RAG+Focus': '3',
    'GPT-4o': '4',
    'GPT-5': '7',
    'Vaccination-clinic physicians': '10',
}

outcome_to_display = {
    'Accuracy': 'Correctness/reliability',
    'Safety': 'Fair and non-misleading',
}


def build_panel_df(panel):
    values = {}
    for row in emm_long.loc[emm_long['panel'].eq(panel)].itertuples(index=False):
        model = source_to_model[row.source]
        dimension = outcome_to_display[row.outcome]
        values[(model, dimension, row.difficulty)] = float(row.adjusted_mean)

    rows = []
    for model in plot_models:
        record = {}
        for dimension in outcome_to_display.values():
            simple = values[(model, dimension, 'Simple')]
            hard = values[(model, dimension, 'Hard')]
            record[(dimension, 'Simple')] = simple
            record[(dimension, 'Hard')] = hard
            record[(dimension, 'Drop (Simple - Hard)')] = simple - hard
        rows.append(record)
    result = pd.DataFrame(rows, index=plot_models)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result


expert_df = build_panel_df('Expert')
parent_df = build_panel_df('Parent')


def plot_panel(ax, df, dimension, title, panel_letter, best_model, hide_xlabel=False):
    y_pos = []
    labels = []
    c_simple = []
    c_hard = []
    c_line = []

    current_y = 0

    for m in tier_human[::-1]:
        y_pos.append(current_y)
        labels.append(model_names_map[m])
        c_simple.append('#CED4DA')
        c_hard.append('#6C757D')
        c_line.append('#E9ECEF')
        current_y += 1

    current_y += 0.5
    ax.axhline(current_y - 0.25, color='#DDDDDD', linestyle='--', linewidth=1.2, zorder=0)

    for m in tier_general[::-1]:
        y_pos.append(current_y)
        labels.append(model_names_map[m])
        c_simple.append('#FFB3C1')
        c_hard.append('#FB6F92')
        c_line.append('#FFE5EC')
        current_y += 1

    current_y += 0.5
    ax.axhline(current_y - 0.25, color='#DDDDDD', linestyle='--', linewidth=1.2, zorder=0)

    for m in tier_fvllm[::-1]:
        y_pos.append(current_y)
        labels.append(model_names_map[m])
        if m == best_model:
            c_simple.append('#A2D2FF')
            c_hard.append('#4361EE')
            c_line.append('#D0D0D0')
        else:
            c_simple.append('#BDE0FE')
            c_hard.append('#8ECAE6')
            c_line.append('#F0F4F8')
        current_y += 1

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color='#F0F0F0', linestyle='-', linewidth=1.2, zorder=0)

    for i, m, y in zip(range(len(plot_models)), plot_models, y_pos):
        try:
            val_simple = df.loc[m, (dimension, 'Simple')]
            val_hard = df.loc[m, (dimension, 'Hard')]
            val_drop = df.loc[m, (dimension, 'Drop (Simple - Hard)')]
        except KeyError:
            continue

        ax.plot([val_hard, val_simple], [y, y], color=c_line[i], zorder=2, linewidth=3.5)
        ax.scatter(val_simple, y, color=c_simple[i], s=140, zorder=4, edgecolor='white', linewidth=1.2)
        ax.scatter(val_hard, y, color=c_hard[i], s=140, zorder=3, edgecolor='white', linewidth=1.2)

        if m == best_model or m in tier_general or m == '10':
            mid_x = (val_simple + val_hard) / 2
            ax.text(mid_x, y + 0.25, f"Δ simple–hard = {val_drop:.2f}", ha='center', va='bottom', fontsize=7.5, color='#555555',
                    fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight='bold', fontsize=9.5)
    ax.tick_params(axis='y', length=0)
    # Reserve a narrow right-hand strip for objective source-group labels.  The
    # former rotated labels were longer than their row groups and visually ran
    # into one another after the figure was down-scaled in Word.
    ax.set_xlim(2.2, 5.52)
    ax.set_xticks([2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    ax.set_title(title, fontweight='bold', fontsize=11, pad=12)
    if not hide_xlabel:
        ax.set_xlabel('Adjusted rating (estimated marginal mean on the 1–5 scale)', fontsize=9.5, fontweight='bold')
    ax.text(-0.25, 1.05, panel_letter, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

    ax.axvline(5.08, color='#D9D9D9', linewidth=0.8)
    ax.text(5.30, y_pos[0], "Human\nresponses", va='center', ha='center', fontsize=6.9,
            color='#6C757D', fontweight='bold', linespacing=0.95)
    ax.text(5.30, np.mean(y_pos[1:3]), "General-purpose\nLLMs", va='center', ha='center', fontsize=6.9,
            color='#FB6F92', fontweight='bold', linespacing=0.95)
    ax.text(5.30, np.mean(y_pos[3:]), "FV-LLM\nconfigurations", va='center', ha='center', fontsize=6.9,
            color='#4361EE', fontweight='bold', linespacing=0.95)


# Assemble the four-panel figure.
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

plt.subplots_adjust(wspace=0.45, hspace=0.35)

# Upper row: the fifth scored item, not clinical safety.
plot_panel(axes[0,0], expert_df, 'Fair and non-misleading', 'Expert: Fair and non-misleading', 'a', best_model='2', hide_xlabel=True)
plot_panel(axes[0,1], parent_df, 'Fair and non-misleading', 'Parent: Fair and non-misleading', 'b', best_model='3', hide_xlabel=True)

# Lower row: human-rated correctness/reliability, not gold-standard factual accuracy.
plot_panel(axes[1,0], expert_df, 'Correctness/reliability', 'Expert: Correctness/reliability', 'c', best_model='2')
plot_panel(axes[1,1], parent_df, 'Correctness/reliability', 'Parent: Correctness/reliability', 'd', best_model='3')

custom_lines = [
    Line2D([0], [0], marker='o', color='white', markerfacecolor='#AAAAAA', markersize=10, linewidth=0,
           label='Simple questions'),
    Line2D([0], [0], marker='o', color='#444444', markerfacecolor='#444444', markersize=10, linewidth=0,
           label='Hard questions')
]
fig.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=False, fontsize=10)

output_pdf = args.out_dir / "Fig3.pdf"
output_png = args.out_dir / "Fig3.png"
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
plt.savefig(output_png, dpi=600, bbox_inches='tight')
print(output_png)
print(output_pdf)
