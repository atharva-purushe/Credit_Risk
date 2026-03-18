import pandas as pd
import numpy as np
import joblib
import os

os.makedirs('.tmp', exist_ok=True)

print("Loading data and model...")
X_test = pd.read_csv('.tmp/X_test.csv')
model = joblib.load('.tmp/model.pkl')

print("Calculating Probability of Default (PD)...")
pd_scores = model.predict_proba(X_test)[:, 1]

# Avoid log(0) mathematically
pd_scores = np.clip(pd_scores, 1e-9, 1 - 1e-9)

print("Calculating FICO-style Scores using PDO method...")
Factor = 20 / np.log(2)
Offset = 600 - Factor * np.log(19)

# Odds of good (1-PD)/PD
odds = (1 - pd_scores) / pd_scores
scores = Offset + Factor * np.log(odds)

# Clip to 300 - 850 range
scores = np.clip(scores, 300, 850)

scorecard = pd.DataFrame({
    'applicant_id': X_test.index,
    'PD': pd_scores,
    'score': np.round(scores).astype(int)
})

print("Assigning risk bands...")
conditions = [
    (scorecard['score'] < 400),
    (scorecard['score'] >= 400) & (scorecard['score'] <= 579),
    (scorecard['score'] >= 580) & (scorecard['score'] <= 669),
    (scorecard['score'] >= 670) & (scorecard['score'] <= 739),
    (scorecard['score'] >= 740)
]
choices = ['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk']
scorecard['risk_band'] = np.select(conditions, choices, default='Unknown')

print("Saving scorecard...")
scorecard.to_csv('.tmp/scorecard.csv', index=False)

print("Updating model_metrics.txt...")
score_mean = scorecard['score'].mean()
score_median = scorecard['score'].median()
score_std = scorecard['score'].std()

band_counts = scorecard['risk_band'].value_counts()
band_pcts = scorecard['risk_band'].value_counts(normalize=True) * 100

metrics_update = [
    "\n=== SCORECARD DISTRIBUTION ===",
    f"Mean Score: {score_mean:.2f}",
    f"Median Score: {score_median:.2f}",
    f"Std Dev: {score_std:.2f}",
    "\n=== RISK BAND DISTRIBUTION ===",
]

for band in ['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk']:
    count = band_counts.get(band, 0)
    pct = band_pcts.get(band, 0)
    metrics_update.append(f"{band:<15s}: {count:>6d} ({pct:>5.1f}%)")

with open('.tmp/model_metrics.txt', 'a') as f:
    f.write('\n' + '\n'.join(metrics_update))

print("Scorecard generation complete.")
