#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
base=pd.read_csv(ROOT/'results/archived/paper_baseline_results.csv')
raw=pd.read_csv(ROOT/'results/archived/repeated_consistency/crc_ton_raw_checkpoint.csv')
summary=pd.read_csv(ROOT/'results/archived/repeated_consistency/crc_ton_f1_stability.csv')

expected_models={'Random Forest','Logistic Regression','Decision Tree','Naive Bayes'}
if set(raw['Seed']) != {42,43,44}: raise SystemExit('FAIL: seed set')
if set(raw['Model']) != expected_models: raise SystemExit('FAIL: model set')
if len(raw) != 12: raise SystemExit(f'FAIL: expected 12 rows, got {len(raw)}')

# Seed 42 detection values must reproduce the archived ToN paper baseline at stored precision.
b=base[(base.Dataset=='ToN_IoT') & base.Model.isin(expected_models)].set_index('Model')
s=raw[raw.Seed==42].set_index('Model')
for model in sorted(expected_models):
    for metric in ['Accuracy','Precision','Recall','F1-score']:
        # Archived paper baseline stores 4 decimals.
        if round(float(s.loc[model,metric]),4) != round(float(b.loc[model,metric]),4):
            raise SystemExit(f'FAIL: seed42 mismatch {model} {metric}: {s.loc[model,metric]} vs {b.loc[model,metric]}')

calc=raw.groupby('Model')['F1-score'].agg(['mean','std']).reset_index()
arch=summary.set_index('Model')
for _,r in calc.iterrows():
    model=r['Model']
    if abs(r['mean']-arch.loc[model,'F1 Mean'])>5e-7 or abs(r['std']-arch.loc[model,'F1 Std'])>5e-7:
        raise SystemExit(f'FAIL: F1 summary mismatch for {model}')

print('PASS: 12 repeated-refit rows present.')
print('PASS: seed-42 detection values reproduce the archived ToN paper baseline at 4-decimal precision.')
print('PASS: archived F1 mean/std values match recomputation from raw repeated-refit rows.')
