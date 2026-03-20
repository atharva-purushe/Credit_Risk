import pandas as pd
import numpy as np
import os

os.makedirs('.tmp', exist_ok=True)

print("Loading data for WoE/IV analysis...")
df = pd.read_csv('data/cs-training.csv', index_col=0)
target_col = 'SeriousDlqin2yrs'

df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

features = df.columns.drop(target_col)
total_events = df[target_col].sum()
total_non_events = len(df) - total_events

woe_data = []
iv_summary = []

for feat in features:
    try:
        df['bin'] = pd.qcut(df[feat], q=10, duplicates='drop')
    except Exception:
        df['bin'] = pd.cut(df[feat], bins=10)
        
    grouped = df.groupby('bin', observed=False).agg(
        events=(target_col, 'sum'),
        total=(target_col, 'count')
    ).reset_index()
    
    grouped['non_events'] = grouped['total'] - grouped['events']
    
    # Avoid dev by zero by adding a small constant
    grouped['dist_events'] = np.maximum(grouped['events'], 0.5) / total_events
    grouped['dist_non_events'] = np.maximum(grouped['non_events'], 0.5) / total_non_events
    
    grouped['woe'] = np.log(grouped['dist_events'] / grouped['dist_non_events'])
    grouped['iv_contribution'] = (grouped['dist_events'] - grouped['dist_non_events']) * grouped['woe']
    
    for _, row in grouped.iterrows():
        woe_data.append({
            'feature': feat,
            'bin_range': str(row['bin']),
            'events': row['events'],
            'non_events': row['non_events'],
            'woe': row['woe'],
            'iv_contribution': row['iv_contribution']
        })
        
    total_iv = grouped['iv_contribution'].sum()
    if total_iv < 0.02: label = "Useless"
    elif total_iv < 0.1: label = "Weak"
    elif total_iv < 0.3: label = "Medium"
    else: label = "Strong"
    
    iv_summary.append({'feature': feat, 'iv': total_iv, 'label': label})

woe_df = pd.DataFrame(woe_data)
woe_df.to_csv('.tmp/woe_iv_table.csv', index=False)

iv_df = pd.DataFrame(iv_summary).sort_values('iv', ascending=False)
iv_text = ["=== INFORMATION VALUE (IV) SUMMARY ==="]
for _, row in iv_df.iterrows():
    iv_text.append(f"{row['feature']:<40} : {row['iv']:.4f} ({row['label']})")

with open('.tmp/iv_summary.txt', 'w') as f:
    f.write('\n'.join(iv_text))

print("WoE and IV computation complete. Results saved to .tmp/")
