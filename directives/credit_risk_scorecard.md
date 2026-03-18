# Credit Risk Scorecard

## Goal
Build a credit risk scorecard that predicts probability of default (PD)
for loan applicants using the Give Me Some Credit dataset.

## Inputs
- `data/cs-training.csv` — 150k loan records, 11 features

## Pipeline Steps
1. EDA — understand distributions, missing values, class imbalance
2. Preprocessing — impute missing, encode, scale features
3. Model training — logistic regression, evaluate AUC
4. Scorecard generation — convert model to interpretable score bands

## Outputs
- `.tmp/eda_report.txt` — EDA summary stats
- `.tmp/model_metrics.txt` — AUC, precision, recall, confusion matrix
- `.tmp/scorecard.csv` — customer-level PD scores + risk bands

## Edge Cases
- SeriousDlqin2yrs is heavily imbalanced (~7% default rate) — use class_weight='balanced'
- MonthlyIncome and NumberOfDependents have significant missing values — impute with median
- Some features have extreme outliers — cap at 99th percentile

## Tools
- execution/eda.py
- execution/preprocess.py
- execution/train_model.py
- execution/generate_report.py