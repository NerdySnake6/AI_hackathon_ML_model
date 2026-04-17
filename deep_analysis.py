"""Deep error analysis on full train.csv."""
import os
os.chdir(r'E:\h_t2')

import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, f1_score
from solution import PredictionModel

df = pd.read_csv('train.csv')
model = PredictionModel()

# Predict in batches
all_preds = []
batch_size = 500
for start in range(0, len(df), batch_size):
    batch = df.iloc[start:start+batch_size]
    preds = model.predict(batch[['QueryText']])
    all_preds.append(preds)
    if (start // batch_size) % 5 == 0:
        print(f'Processed {min(start+batch_size, len(df))}/{len(df)}', flush=True)

preds = pd.concat(all_preds, ignore_index=True)
preds.to_csv('full_preds.csv', index=False, encoding='utf-8')

# ====== METRICS ======
print('\n========== METRICS ==========')

# TypeQuery F2
tq_f2 = fbeta_score(df['TypeQuery'], preds['TypeQuery'], beta=2)
print(f'TypeQuery F2: {tq_f2:.4f}')

# False positives / negatives
mask_true = df['TypeQuery'] == 1
mask_pred = preds['TypeQuery'] == 1

fp = df[(df['TypeQuery'] == 0) & (preds['TypeQuery'] == 1)]
fn = df[(df['TypeQuery'] == 1) & (preds['TypeQuery'] == 0)]
print(f'  False Positives: {len(fp)} / {len(df[df["TypeQuery"]==0])}')
print(f'  False Negatives: {len(fn)} / {len(df[df["TypeQuery"]==1])}')

# ContentType macro F1
ct_true = df.loc[mask_true, 'ContentType'].fillna('').astype(str).values
ct_pred = preds.loc[mask_true, 'ContentType'].fillna('').astype(str).values
labels = ['сериал', 'фильм', 'мультфильм', 'мультсериал', 'прочее']
ct_f1 = f1_score(ct_true, ct_pred, average='macro', labels=labels, zero_division=0)
print(f'\nContentType macro F1: {ct_f1:.4f}')

# Per-class F1
for lab in labels:
    t = (ct_true == lab).astype(int)
    p = (ct_pred == lab).astype(int)
    if t.sum() == 0 and p.sum() == 0:
        continue
    f = f1_score(t, p, zero_division=0)
    print(f'  {lab:<15s} F1={f:.3f} (true={t.sum()}, pred={p.sum()})')

# Title token F1
scores = []
exact_match = 0
partial_match = 0
total = 0
for i in range(len(df)):
    if df.iloc[i]['TypeQuery'] != 1:
        continue
    total += 1
    t_tokens = set(str(df.iloc[i].get('Title', '')).lower().split())
    p_tokens = set(str(preds.iloc[i]['Title']).lower().split())
    if t_tokens == p_tokens and t_tokens:
        exact_match += 1
    if not t_tokens and not p_tokens:
        scores.append(1.0)
        continue
    if not t_tokens or not p_tokens:
        scores.append(0.0)
        continue
    inter = t_tokens & p_tokens
    prec = len(inter) / len(p_tokens) if p_tokens else 0
    rec = len(inter) / len(t_tokens) if t_tokens else 0
    f = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
    scores.append(f)
    if f > 0 and f < 1.0:
        partial_match += 1

title_f1 = np.mean(scores) if scores else 0
print(f'\nTitle token F1: {title_f1:.4f}')
print(f'  Exact matches: {exact_match}/{total} ({100*exact_match/total:.1f}%)')
print(f'  Partial matches: {partial_match}/{total} ({100*partial_match/total:.1f}%)')
empty_both = sum(1 for s in scores if s == 1.0)  # both empty = score 1.0
print(f'  Both empty (score=1.0): {empty_both}/{total} ({100*empty_both/total:.1f}%)')

combined = 0.35*tq_f2 + 0.30*ct_f1 + 0.35*title_f1
print(f'\nCombined: {combined:.4f}')

# ====== ERROR ANALYSIS ======
print('\n========== TYPEQUERY FALSE POSITIVES ==========')
print(f'Total: {len(fp)}')
for _, row in fp.head(10).iterrows():
    print(f'  Q: {row["QueryText"][:70]}')

print('\n========== TYPEQUERY FALSE NEGATIVES ==========')
print(f'Total: {len(fn)}')
for _, row in fn.head(10).iterrows():
    print(f'  Q: {row["QueryText"][:70]}')

print('\n========== TITLE ERRORS (TypeQuery=1, has true title) ==========')
has_title = df[(df['TypeQuery']==1) & (df['Title'].notna()) & (df['Title']!='')]
wrong_title = has_title[has_title['Title'].str.lower() != preds.loc[has_title.index, 'Title'].str.lower()]
print(f'Wrong titles: {len(wrong_title)}/{len(has_title)}')

# Common error patterns
error_queries = wrong_title.head(50)
for _, row in error_queries.iterrows():
    idx = row.name
    true_t = row['Title']
    pred_t = preds.loc[idx, 'Title']
    print(f'  Q: {row["QueryText"][:60]}')
    print(f'    true="{true_t}" pred="{pred_t}"')
    print()
