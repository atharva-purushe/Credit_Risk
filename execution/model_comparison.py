import pandas as pd
import numpy as np
import time
import os
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import ks_2samp
from xgboost import XGBClassifier

os.makedirs('.tmp', exist_ok=True)

print("Loading data for model comparison...")
X_train = pd.read_csv('.tmp/X_train.csv')
X_test = pd.read_csv('.tmp/X_test.csv')
y_train = pd.read_csv('.tmp/y_train.csv')['SeriousDlqin2yrs']
y_test = pd.read_csv('.tmp/y_test.csv')['SeriousDlqin2yrs']

def get_metrics(model, name, X_train, y_train, X_test, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    t_time = time.time() - start
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    ks_stat, _ = ks_2samp(y_prob[y_test == 0], y_prob[y_test == 1])
    
    return {
        'Model': name, 'Train_Time_sec': t_time, 
        'ROC_AUC': auc, 'KS_Stat': ks_stat * 100,
        'Precision': prec, 'Recall': rec, 'F1': f1
    }

print("Training Model A: Calibrated Logistic Regression...")
lr_base = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model = CalibratedClassifierCV(lr_base, method='isotonic', cv=5)
lr_res = get_metrics(lr_model, 'Logistic Regression', X_train, y_train, X_test, y_test)

print("Training Model B: XGBoost...")
ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = XGBClassifier(scale_pos_weight=ratio, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_res = get_metrics(xgb_model, 'XGBoost', X_train, y_train, X_test, y_test)

res_df = pd.DataFrame([lr_res, xgb_res]).set_index('Model').T

out_text = []
out_text.append("=== MODEL COMPARISON ===")
out_text.append(res_df.to_string())
out_text.append("\n=== CONCLUSION ===")
out_text.append(
    "While XGBoost may yield a slightly higher ROC AUC due to its ability to model complex non-linear mathematical interactions "
    "without any feature transformations, Logistic Regression remains the strongly preferred choice for this credit risk scorecard. "
    "XGBoost is a 'black-box' model, making it incredibly difficult to explain individual score decisions to regulators "
    "(such as for printing Adverse Action notices for loans under the FCRA). In contrast, Logistic Regression provides direct, interpretable "
    "linear coefficients that map perfectly into the Proportional to Odds (PDO) scorecard methodology. "
    "This interpretability, strict regulatory compliance, and auditability far outweigh the marginal predictive lift provided by XGBoost."
)

with open('.tmp/model_comparison.txt', 'w') as f:
    f.write('\n'.join(out_text))

print("Model comparison complete. Results saved to .tmp/")
