# FV-LLM analysis and figure code

This repository contains the statistical analysis and code-generated figure
workflows supporting the FV-LLM manuscript. It starts from the frozen,
analysis-ready rating dataset described in the manuscript and reproduces the
primary mixed-effects models, reported sensitivity analyses, descriptive
summaries and code-generated figures.

## Repository scope

Included:

- primary and response-length-adjusted mixed-effects models;
- principal component analysis and response-length summaries;
- difficulty-profile models used for Fig. 3 and Supplementary Fig. S4;
- scripts for Figs. 1–3 and 5 and Supplementary Figs. S1–S4;
- aggregate result files and reference figure outputs.

Not included:

- participant-level rating data, response text or other restricted research
  data;
- the deployed chatbot source code, model-provider API wrappers or the
  retrieval index;
- Fig. 4, which is a system-architecture schematic, or Supplementary Fig. S5,
  which contains interface screenshots.

This scope matches the manuscript's **analysis code** availability statement;
it should not be interpreted as a release of the full FV-LLM platform.

## Repository structure

```text
data/                       Data-access instructions, data dictionary and source codebook
scripts/analysis/           Statistical analyses and derived summaries
scripts/figures/            Code-generated manuscript and supplementary figures
results/                    Aggregate reference results reported in the manuscript
outputs/main_figures/       Reference outputs for code-generated main figures
outputs/supplementary_figures/
requirements.txt            Version-pinned Python dependencies
run_all.py                  Reproduce analyses and figures from the analysis-ready dataset
```

Historical scripts, internal review notes, Word-editing utilities and
intermediate plotting scripts are intentionally excluded.

## Manuscript output map

| Manuscript output | Analysis script | Figure script or aggregate output |
|---|---|---|
| Primary mixed-effects results (Table 2 and Supplementary Table S4) | `scripts/analysis/run_primary_models.py` | `results/primary/` |
| Response-length-adjusted sensitivity analyses (Supplementary Tables S5–S6) | `scripts/analysis/run_length_adjusted_models.py` | `results/length_adjusted/` |
| PCA results (Supplementary Table S2 and Fig. S1) | `scripts/analysis/run_pca.py` | `scripts/figures/figs1_pca.py` |
| Response-length summary (Supplementary Table S3) | `scripts/analysis/summarize_response_lengths.py` | `results/descriptive/response_length_summary.csv` |
| Standardized rating profiles (Figs. 1, 2 and S2) | `scripts/analysis/summarize_standardized_scores.py` | `scripts/figures/fig1_standardized_scores.py`, `scripts/figures/fig2_figs2_radar.py` |
| Difficulty profiles (Fig. 3) | `scripts/analysis/run_fig3_difficulty_models.py` | `scripts/figures/fig3_difficulty_profiles.py` |
| Primary model contrasts (Fig. S3) | `scripts/analysis/run_primary_models.py` | `scripts/figures/figs3_primary_contrasts.py` |
| All-source difficulty profiles (Fig. S4) | `scripts/analysis/run_figs4_difficulty_models.py` | `scripts/figures/figs4_difficulty_profiles.py` |
| Evaluation design (Fig. 5) | Not applicable | `scripts/figures/fig5_evaluation_design.py` |

## Data access

The participant-level analysis dataset is not distributed in this repository.
Qualified researchers may request access from the corresponding author,
subject to ethics approval, participant-consent constraints, institutional
review and an appropriate data-use agreement. See [`data/README.md`](data/README.md)
for the expected filename and variable definitions.

After access is approved, place the following file locally:

```text
data/analysis_dataset.csv
```

The file is ignored by Git and must not be committed.

## Software environment

The analyses were conducted in Python 3.9. Install the version-pinned
dependencies in an isolated environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducing the analyses

Run the complete workflow from the repository root:

```bash
python run_all.py --data data/analysis_dataset.csv
```

The command writes analysis results under `results/` and regenerates all
code-generated figures under `outputs/`. Existing aggregate results are
provided as reference outputs so that figure rendering can be checked without
access to participant-level data.

Individual scripts expose command-line help, for example:

```bash
python scripts/analysis/run_primary_models.py --help
python scripts/figures/fig3_difficulty_profiles.py --help
```

## Analysis specifications

For each of the six 1–5 rating items, the primary model included response
source, evaluator panel, their interaction and question difficulty as fixed
effects. Random intercepts were specified for evaluator, question and the
question-by-source response. The sensitivity model additionally adjusted for
response length per 100 cleaned Chinese characters. RAG and RAG+Focus were
contrasted with vaccination-clinic physicians within each evaluator panel;
Holm correction was applied within the multiplicity families defined in the
manuscript.

The frozen analysis-ready dataset contains 2,600 complete ratings for each
outcome: 1,300 expert-panel ratings and 1,300 parent-panel ratings. It contains
one record for every panel–question–source combination. `run_all.py` validates
this complete 2 × 100 × 13 allocation grid and the response metadata before
fitting any model. No outcome value is imputed during the analysis workflow.

## Licence

The source code is released under the [MIT License](LICENSE). This software
licence does not grant permission to redistribute restricted study data,
response text, third-party content or model outputs.
