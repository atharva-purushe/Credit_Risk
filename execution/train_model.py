import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

os.makedirs('.tmp', exist_ok=True)

print("Loading preprocessed data...")
X_train = pd.read_csv('.tmp/X_train.csv')
X_test = pd.read_csv('.tmp/X_test.csv')
y_train = pd.read_csv('.tmp/y_train.csv')['SeriousDlqin2yrs']
y_test = pd.read_csv('.tmp/y_test.csv')['SeriousDlqin2yrs']

print("Training Calibrated Logistic Regression model...")
base_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

metrics_out = []
metrics_out.append("=== MODEL EVALUATION METRICS ===")
metrics_out.append(f"ROC AUC Score: {auc:.4f}\n")
metrics_out.append("Classification Report:")
metrics_out.append(report)
metrics_out.append("\nConfusion Matrix:")
metrics_out.append(str(conf_matrix))

output_str = '\n'.join(metrics_out)
with open('.tmp/model_metrics.txt', 'w') as f:
    f.write(output_str)

print("Saving model artifacts...")
joblib.dump(model, '.tmp/model.pkl')

print("Model training complete. Metrics saved to .tmp/model_metrics.txt")
