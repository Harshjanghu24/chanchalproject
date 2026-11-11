# CDAS package: AI anxiety and cognitive dependence

Start with the **results summary** (PDF or Word). This README says what the code does and how to run it.

Everything here reports **associations, not causes**. AI anxiety is a research measure, not a diagnosis, and no clinical cut-offs are used.

## What is in the package

```
README.md
results_summary.pdf (or .docx)
part1_public_datasets/   analysis of existing public student surveys
part2_cogsense/          pipeline + dashboard for the AI-CogSense (AIAS) survey
```

| Part | Purpose | Status |
|---|---|---|
| Part 1 | Random Forest, Bayesian network, Ridge/ElasticNet and XGBoost on four public datasets | Run on real public data; results are in the summary |
| Part 2 | Cleaning/scoring, statistics, ML and a Streamlit dashboard for the AIAS survey | Tested on **fake data only**; waiting for real survey responses |

## Setup (once)

Python 3.12 is recommended.

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
pip install pandas numpy scikit-learn scipy statsmodels matplotlib joblib xgboost pgmpy networkx openpyxl streamlit
```

Optional: `pip install shap` for SHAP plots in Part 2.

## Part 1: public datasets

Download the data yourself (not included, to respect each licence; check the terms on each page):

| File to save in `part1_public_datasets/datasets/` | Source |
|---|---|
| `AI_Literacy_questionnaire_data_.csv` | https://figshare.com/articles/dataset/AI_Literacy_Questionnaire_data/29488523/2 |
| `Career_Anxiety_due_to_AI.xlsx` | https://data.mendeley.com/datasets/r3xh8sd764/2 |
| `Data_set_Ex.csv` (cognitive strain and emotional responses to AI) | https://data.mendeley.com/datasets/9v9rtzch8g/2 |
| the cognitive dependence CSV (file name contains `AI_Cognitive_Dependence`) | https://data.mendeley.com/datasets/7m94b7yr8w/2 |

Run from inside `part1_public_datasets/`, in this order:

```
python data_prep.py          # builds scale scores; prints sample sizes and correlations
python train_models.py       # Ridge / Random Forest / XGBoost on the Figshare data -> results/
python cv_eval.py            # repeated user-grouped CV, a more stable comparison
python bayes_net.py          # Bayesian network -> results/bn_graph.png
python multi_datasets.py     # the three other datasets, incl. benefit x cost analysis
```

If a file lives elsewhere, set `CDAS_FIGSHARE_CSV` to its path, or pass the datasets folder to `multi_datasets.py` as an argument.

Key design choices:
- Anxiety items (A1-A5) are the target and are never used as predictors.
- The Figshare rows are split by student ID, so a repeated student never appears in both training and test data.
- Out-of-sample R-squared is compared against a train-mean baseline. Values near zero mean no useful prediction.
- In the Figshare data, 0 codes in the course-related scales (IM, S, C, BI) are missing values, not low scores.

## Part 2: AI-CogSense (AIAS survey)

Files: `cogsense_pipeline.py`, `cogsense_dashboard.py`, `make_demo_data.py`. Run from inside `part2_cogsense/`.

Try it on fake data:

```
python make_demo_data.py
python cogsense_pipeline.py cogsense_demo_responses.csv cogsense_results
streamlit run cogsense_dashboard.py
```

The dashboard shows a **DEMO DATA** banner for files with "demo" in the name. Fake data must never be used as a result.

With real responses:
1. Export the Google Form responses to CSV.
2. Rename the columns exactly as in the Data_Dictionary sheet (`participant_id`, `group`, `age_group`, ..., `C01`...`C19`, `AIAS01`...`AIAS21`). The script stops with a clear message if a column is missing.
3. Run `python cogsense_pipeline.py responses.csv cogsense_results`, then the dashboard.

What it does: removes duplicate and incomplete rows and flags straight-lining; scores AIAS total and its four dimensions as means; compares students and professionals (Mann-Whitney, with FDR adjustment); correlates concern and dependence scores with AIAS; trains a baseline, ElasticNet and Random Forest, with a model that leaves out C01-C03 as a check against overlap with the AIAS job-replacement factor. Groups under 5 people are hidden.

Sample size guard: under 60 responses the ML step is skipped. A 15-30 person pilot is for checking the form, not for modelling.

## Limits to keep in mind

- The public samples are students only. Students versus professionals, privacy concern, locus of control and imposter feelings could not be tested.
- Some strong results (for example, regret and fear of weaker thinking) come from near-duplicate questions, not independent constructs.
- The AIAS items must be obtained and used under the authors' or publisher's terms; this package does not contain them.
- Do not collect or store names, emails or phone numbers.