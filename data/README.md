# Restricted analysis data

The participant-level rating dataset is not publicly distributed. Access may
be requested from the corresponding author and is subject to ethics approval,
participant-consent constraints, institutional review and an appropriate
data-use agreement.

After access is approved, save the analysis-ready file as:

```text
data/analysis_dataset.csv
```

The analysis scripts expect one row for each panel–question–source rating and
the following variables:

| Variable | Description |
|---|---|
| `question_id` | Evaluation-question identifier (1–100) |
| `group` | Numeric response-source code |
| `panel` | Evaluator panel (`Expert` or `Parent`) |
| `rater` | Pseudonymous evaluator identifier |
| `Q1OEAE` | Correctness/reliability rating (1–5) |
| `Q2OEAA` | Clarity rating (1–5) |
| `Q3OEAA` | Completeness rating (1–5) |
| `Q4OEOEOE` | Empathy rating (1–5) |
| `Q5OEAA` | Fair and non-misleading rating (1–5) |
| `Q6OEAEOEOE` | Perceived absence of bias rating (1–5) |
| `response_length` | Cleaned response length in Chinese characters |
| `difficulty` | Question difficulty (`Simple` or `Hard`) |
| `response_id` | Question-by-source response identifier |
| `group_code` | String/numeric response-source code used by the models |
| `length_100` | `response_length / 100` |

The source labels corresponding to `group_code` are provided in
[`source_codebook.csv`](source_codebook.csv).

The final analysis-ready file contains 2,600 rows and complete values for all
six outcomes: 1,300 expert-panel ratings and 1,300 parent-panel ratings. Every
combination in the 2 panels × 100 questions × 13 response sources appears
exactly once. `scripts/data/validate_analysis_dataset.py` checks this structure,
the rating ranges and the response metadata before any model is fitted.

The file is excluded by `.gitignore`; do not commit it to a public repository.
