#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, os, threading, time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

SEED = 42
SAMPLE_INTERVAL = 0.05

def monitor(stop, peaks):
    proc = psutil.Process(os.getpid())
    while not stop.is_set():
        peaks.append(proc.memory_info().rss / (1024**2))
        time.sleep(SAMPLE_INTERVAL)

def evaluate(name, model, Xtr, ytr, Xte, yte):
    gc.collect()
    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss / (1024**2)
    peaks = [before]
    stop = threading.Event()
    th = threading.Thread(target=monitor, args=(stop, peaks), daemon=True)
    th.start()
    t0 = time.perf_counter(); model.fit(Xtr, ytr); train_s = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(Xte); infer_s = time.perf_counter() - t0
    stop.set(); th.join(timeout=1)
    memory_mb = max(0.0, max(peaks) - before)
    return {
        "Dataset":"UNSW-NB15", "Model":name,
        "Accuracy":accuracy_score(yte,pred), "Precision":precision_score(yte,pred,zero_division=0),
        "Recall":recall_score(yte,pred,zero_division=0), "F1-score":f1_score(yte,pred,zero_division=0),
        "Training Time (s)":train_s, "Inference Time (s)":infer_s,
        "Memory Usage (MB)":memory_mb, "Test Samples":len(yte),
        "Throughput (samples/s)":len(yte)/infer_s if infer_s>0 else float("inf"),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project_root', required=True)
    p.add_argument('--out_dir', default='results/generated/unsw_baseline')
    p.add_argument('--skip_svm', action='store_true', help='Skip the very expensive SVM baseline.')
    a=p.parse_args(); root=Path(a.project_root).expanduser().resolve()
    tr=root/'data/raw/unsw_nb15/UNSW_NB15_training-set.csv'; te=root/'data/raw/unsw_nb15/UNSW_NB15_testing-set.csv'
    if not tr.exists() or not te.exists(): raise FileNotFoundError(f'Missing UNSW-NB15 files: {tr} / {te}')
    dtr=pd.read_csv(tr); dte=pd.read_csv(te); target='label'; drops=['id','attack_cat',target]
    Xtr=dtr.drop(columns=drops,errors='ignore'); ytr=dtr[target].astype(int)
    Xte=dte.drop(columns=drops,errors='ignore'); yte=dte[target].astype(int)
    cats=Xtr.select_dtypes(include=['object','string','category']).columns.tolist()
    Xtr=pd.get_dummies(Xtr,columns=cats); Xte=pd.get_dummies(Xte,columns=cats)
    Xtr,Xte=Xtr.align(Xte,join='left',axis=1,fill_value=0)
    Xtr=Xtr.astype(np.float32); Xte=Xte.astype(np.float32)
    models=[('Random Forest',RandomForestClassifier(n_estimators=100,random_state=SEED,n_jobs=-1)),
            ('MLP',MLPClassifier(hidden_layer_sizes=(64,),max_iter=100,random_state=SEED)),
            ('Logistic Regression',LogisticRegression(max_iter=2000,solver='liblinear',random_state=SEED))]
    if not a.skip_svm: models.insert(1,('SVM',SVC()))
    rows=[]
    for name,m in models:
        print(f'Running {name}...', flush=True); rows.append(evaluate(name,m,Xtr,ytr,Xte,yte))
    out=(root/a.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    path=out/'unsw_nb15_baseline_generated.csv'; pd.DataFrame(rows).to_csv(path,index=False); print(path)
if __name__=='__main__': main()
