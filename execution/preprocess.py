import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs('.tmp', exist_ok=True)

print("Loading data...")
df = pd.read_csv('data/cs-training.csv', index_col=0)

print("Imputing missing values with median...")
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

print("Capping outliers at 99th percentile...")
target_col = 'SeriousDlqin2yrs'
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

for col in numeric_cols:
    p99 = df[col].quantile(0.99)
    df[col] = np.where(df[col] > p99, p99, df[col])

X = df.drop(columns=[target_col])
y = df[target_col]

print("Splitting dataset into train and test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("Saving preprocessed artifacts to .tmp/ ...")
X_train_scaled.to_csv('.tmp/X_train.csv', index=False)
X_test_scaled.to_csv('.tmp/X_test.csv', index=False)
y_train.to_csv('.tmp/y_train.csv', index=False)
y_test.to_csv('.tmp/y_test.csv', index=False)

print("Preprocessing complete.")
