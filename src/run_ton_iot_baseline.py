#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, os, tempfile, threading, time
from pathlib import Path
import joblib
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

SEED=42; SAMPLE_INTERVAL=0.05; DROP=['label','type','src_ip','dst_ip']

def monitor(stop, peaks):
    proc=psutil.Process(os.getpid())
    while not stop.is_set(): peaks.append(proc.memory_info().rss/(1024**2)); time.sleep(SAMPLE_INTERVAL)

def evaluate(name, model, Xtr,ytr,Xte,yte):
    gc.collect(); proc=psutil.Process(os.getpid()); before=proc.memory_info().rss/(1024**2)
    peaks=[before]; stop=threading.Event(); th=threading.Thread(target=monitor,args=(stop,peaks),daemon=True); th.start()
    t0=time.perf_counter(); model.fit(Xtr,ytr); train_s=time.perf_counter()-t0
    t0=time.perf_counter(); pred=model.predict(Xte); infer_s=time.perf_counter()-t0
    stop.set(); th.join(timeout=1); memory=max(0.0,max(peaks)-before)
    with tempfile.TemporaryDirectory() as td:
        mp=Path(td)/'model.joblib'; joblib.dump(model,mp); model_size=mp.stat().st_size/(1024**2)
    return {'Dataset':'ToN_IoT','Model':name,'Accuracy':accuracy_score(yte,pred),
            'Precision':precision_score(yte,pred,zero_division=0),'Recall':recall_score(yte,pred,zero_division=0),
            'F1-score':f1_score(yte,pred,zero_division=0),'Training Time (s)':train_s,'Inference Time (s)':infer_s,
            'Memory Usage (MB)':memory,'Test Samples':len(yte),'Throughput (samples/s)':len(yte)/infer_s if infer_s>0 else float('inf'),
            'Model Size (MB)':model_size}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--project_root',required=True); p.add_argument('--out_dir',default='results/generated/ton_iot_baseline')
    a=p.parse_args(); root=Path(a.project_root).expanduser().resolve(); fp=root/'data/raw/ton_iot/train_test_network.csv'
    if not fp.exists(): raise FileNotFoundError(fp)
    df=pd.read_csv(fp); missing=[c for c in DROP if c not in df.columns]
    if missing: raise ValueError(f'Missing columns: {missing}')
    y=df['label'].astype(int); X=df.drop(columns=DROP); cats=X.select_dtypes(include=['object','string']).columns.tolist()
    X=pd.get_dummies(X,columns=cats)  # conference protocol: no explicit float32 coercion
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)
    models=[('Random Forest',RandomForestClassifier(n_estimators=100,random_state=SEED,n_jobs=-1)),
            ('Logistic Regression',LogisticRegression(max_iter=2000,solver='liblinear',random_state=SEED)),
            ('MLP',MLPClassifier(hidden_layer_sizes=(64,),max_iter=100,random_state=SEED)),
            ('Decision Tree',DecisionTreeClassifier(random_state=SEED)),('Naive Bayes',GaussianNB())]
    rows=[]
    for name,m in models:
        print(f'Running {name}...',flush=True); rows.append(evaluate(name,m,Xtr,ytr,Xte,yte))
    out=(root/a.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True); path=out/'ton_iot_baseline_generated.csv'
    pd.DataFrame(rows).to_csv(path,index=False); print(path)
if __name__=='__main__': main()
