import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

os.makedirs('.tmp', exist_ok=True)

print("Loading data and model...")
X_test = pd.read_csv('.tmp/X_test.csv')
X_train = pd.read_csv('.tmp/X_train.csv')
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

print("Computing SHAP values...")
try:
    # CalibratedClassifierCV breaks LinearExplainer, so we extract the base logistic model safely
    if hasattr(model, 'calibrated_classifiers_'):
        explainer_model = getattr(model.calibrated_classifiers_[0], 'estimator', None)
        if explainer_model is None:
            explainer_model = model.calibrated_classifiers_[0].base_estimator
    else:
        explainer_model = model

    explainer = shap.LinearExplainer(explainer_model, X_train)
    shap_values = explainer(X_test)
except Exception as e:
    print(f"Fallback due to calibration wrapper: {e}")
    # Using independent Explainer
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_test)

print("Processing SHAP summaries...")
# Calculate mean absolute SHAP over the test set
mean_shap = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({'Feature': X_test.columns, 'Mean_Abs_SHAP': mean_shap})
shap_df = shap_df.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

shap_out = ["=== SHAP FEATURE IMPORTANCE ==="]
for idx, row in shap_df.iterrows():
    shap_out.append(f"{idx+1}. {row['Feature']:<30} : {row['Mean_Abs_SHAP']:.4f}")

shap_out.append("\n=== TOP 3 FACTORS DRIVING DEFAULT RISK ===")
for idx in range(3):
    shap_out.append(f"- {shap_df.iloc[idx]['Feature']}")

with open('.tmp/shap_summary.txt', 'w') as f:
    f.write('\n'.join(shap_out))

print("Plotting SHAP summary...")
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('.tmp/shap_summary.png', bbox_inches='tight')
plt.close()

print("Scorecard generation and SHAP analysis complete.")
