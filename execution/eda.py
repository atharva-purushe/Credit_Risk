# eda.py — Exploratory Data Analysis for Credit Risk Dataset
# Input: data/cs-training.csv
# Output: .tmp/eda_report.txt

import pandas as pd
import numpy as np
import os

os.makedirs('.tmp', exist_ok=True)

df = pd.read_csv('data/cs-training.csv', index_col=0)

report = []

report.append("=== BASIC INFO ===")
report.append(f"Shape: {df.shape}")
report.append(f"\nColumns:\n{list(df.columns)}")
report.append(f"\nDtypes:\n{df.dtypes.to_string()}")

report.append("\n=== MISSING VALUES ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
report.append(pd.DataFrame({'missing': missing, 'pct': missing_pct}).to_string())

report.append("\n=== TARGET DISTRIBUTION ===")
target = df['SeriousDlqin2yrs']
report.append(f"Default rate: {target.mean()*100:.2f}%")
report.append(f"Class counts:\n{target.value_counts().to_string()}")

report.append("\n=== DESCRIPTIVE STATS ===")
report.append(df.describe().to_string())

report.append("\n=== OUTLIER CHECK (99th percentile) ===")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove('SeriousDlqin2yrs')
for col in numeric_cols:
    p99 = df[col].quantile(0.99)
    outliers = (df[col] > p99).sum()
    report.append(f"{col}: 99th pct = {p99:.2f}, outliers above = {outliers}")

output = '\n'.join(report)
with open('.tmp/eda_report.txt', 'w') as f:
    f.write(output)

print("EDA complete. Report saved to .tmp/eda_report.txt")
print(output)