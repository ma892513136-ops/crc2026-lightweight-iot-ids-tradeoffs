#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CRC 2026 — ToN_IoT repeated-refit consistency runner

Default scope:
- ToN_IoT network subset only
- Exact CRC preprocessing mirror: drop label/type/src_ip/dst_ip; get_dummies before split;
  no float32 coercion
- Seeds 42, 43, 44
- Default models: Random Forest, Logistic Regression, Decision Tree, Naive Bayes
- MLP excluded by default because it is extremely expensive under the exact-CRC dtype path

Safety:
- Writes only to the requested generated-output directory
- Checkpoints immediately after each completed seed/model
- Resumable: completed seed/model pairs are skipped
- Rejects sklearn results that report 'Training interrupted by user'
- Does not modify archived paper results
- No memory/CPU profiling; timings are diagnostic only
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_MODELS = ["Random Forest", "Logistic Regression", "Decision Tree", "Naive Bayes"]
ALL_MODELS = DEFAULT_MODELS + ["MLP"]
DROP_COLS = ["label", "type", "src_ip", "dst_ip"]


def parse_args():
    p = argparse.ArgumentParser(description="Resumable exact-CRC ToN_IoT repeated-refit consistency check.")
    p.add_argument("--project_root", required=True)
    p.add_argument(
        "--out_dir",
        default="results/generated/repeated_consistency",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    return p.parse_args()


def resolve_paths(args):
    root = Path(args.project_root).expanduser().resolve()
    ton = root / "data" / "raw" / "ton_iot" / "train_test_network.csv"
    if not root.exists():
        raise FileNotFoundError(root)
    if not ton.exists():
        raise FileNotFoundError(ton)

    out_arg = Path(args.out_dir).expanduser()
    out = out_arg.resolve() if out_arg.is_absolute() else (root / out_arg).resolve()

    protected = (root / "results" / "archived").resolve()
    if out == protected or protected in out.parents:
        raise RuntimeError(f"Refusing to write inside archived paper results: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return root, ton, out


def load_ton(ton_path):
    df = pd.read_csv(ton_path)
    missing = [c for c in DROP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    y = df["label"].astype(int)
    X = df.drop(columns=DROP_COLS)
    cats = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X = pd.get_dummies(X, columns=cats)  # exact CRC mirror; no float32 cast
    info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "removed_columns": DROP_COLS,
        "categorical_columns": cats,
        "encoded_feature_dimension": int(X.shape[1]),
        "encoding_policy": "pandas.get_dummies before split; no float32 coercion",
    }
    return X, y, info


def make_model(name, seed):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "Naive Bayes":
        return GaussianNB()
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=100, random_state=seed)
    raise ValueError(name)


def atomic_csv(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_raw(path):
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    required = {"Dataset", "Seed", "Model", "Accuracy", "Precision", "Recall", "F1-score"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"Checkpoint schema mismatch: {sorted(missing)}")
    return raw


def pairs(raw):
    if raw.empty:
        return set()
    return {(int(r.Seed), str(r.Model)) for r in raw.itertuples(index=False)}


def evaluate(seed, model_name, model, Xtr, Xte, ytr, yte):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        train_s = time.perf_counter() - t0
    msgs = [str(w.message) for w in caught]
    if any("Training interrupted by user" in m for m in msgs):
        raise RuntimeError("Training interrupted by user; result rejected")

    t0 = time.perf_counter()
    pred = model.predict(Xte)
    infer_s = time.perf_counter() - t0

    return {
        "Dataset": "ToN_IoT",
        "Seed": int(seed),
        "Model": model_name,
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred, zero_division=0),
        "Recall": recall_score(yte, pred, zero_division=0),
        "F1-score": f1_score(yte, pred, zero_division=0),
        "Training Time Diagnostic (s)": train_s,
        "Inference Time Diagnostic (s)": infer_s,
        "Train Samples": int(len(ytr)),
        "Test Samples": int(len(yte)),
        "Feature Count": int(Xtr.shape[1]),
        "Warnings": " | ".join(msgs),
    }


def summaries(raw, summary_path, f1_path):
    if raw.empty:
        return
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    s = raw.groupby(["Dataset", "Model"])[metrics].agg(["count", "mean", "std", "min", "max"]).reset_index()
    s.columns = [" ".join([str(x) for x in c if str(x)]).strip() if isinstance(c, tuple) else str(c) for c in s.columns]
    atomic_csv(s.round(6), summary_path)

    f = raw.groupby("Model")["F1-score"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    f = f.rename(columns={"count":"Completed Seeds","mean":"F1 Mean","std":"F1 Std","min":"F1 Min","max":"F1 Max"})
    f["F1 Range"] = f["F1 Max"] - f["F1 Min"]
    f = f.sort_values("F1 Mean", ascending=False).reset_index(drop=True)
    f.insert(0, "Current Mean-F1 Rank", np.arange(1, len(f)+1))
    atomic_csv(f.round(6), f1_path)


def write_protocol(path, ton_path, args, info):
    payload = {
        "purpose": "CRC 2026 major revision — ToN_IoT repeated-refit consistency",
        "input_file": str(ton_path),
        "requested_seeds": [int(s) for s in args.seeds],
        "requested_models": list(args.models),
        "preprocessing": info,
        "split_policy": "80:20 stratified split repeated independently for each seed",
        "checkpoint_policy": "atomic raw checkpoint after every completed seed/model",
        "resume_policy": "completed seed/model pairs skipped",
        "interruption_policy": "sklearn 'Training interrupted by user' results are rejected",
        "statistical_boundary": "descriptive mean/std only; no inferential significance claim",
        "resource_boundary": "timings diagnostic only; frozen CRC resource values remain authoritative",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    root, ton_path, out = resolve_paths(args)
    raw_path = out / "crc_ton_raw_checkpoint.csv"
    summary_path = out / "crc_ton_detection_summary.csv"
    f1_path = out / "crc_ton_f1_stability.csv"
    protocol_path = out / "crc_ton_protocol.json"

    print("=== CRC 2026: ToN_IoT repeated-refit consistency ===")
    print(f"Project root: {root}")
    print(f"Input:        {ton_path}")
    print(f"Output:       {out}")
    print(f"Seeds:        {args.seeds}")
    print(f"Models:       {args.models}")
    print("Checkpoint after every completed seed/model; completed pairs are skipped.\n")

    X, y, info = load_ton(ton_path)
    if not protocol_path.exists():
        write_protocol(protocol_path, ton_path, args, info)

    raw = load_raw(raw_path)
    done = pairs(raw)

    for seed in args.seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        print(f"--- seed={seed} ---")
        for model_name in args.models:
            key = (int(seed), model_name)
            if key in done:
                print(f"SKIP completed: {model_name}")
                continue
            print(f"Running {model_name}...", flush=True)
            try:
                result = evaluate(seed, model_name, make_model(model_name, seed), Xtr, Xte, ytr, yte)
            except KeyboardInterrupt:
                print(f"\nInterrupted during {model_name} seed={seed}; no result checkpointed.")
                raise
            except Exception as exc:
                print(f"ERROR: {model_name} seed={seed}: {exc}")
                print("This pair was NOT checkpointed.")
                return

            raw = pd.concat([raw, pd.DataFrame([result])], ignore_index=True)
            raw = raw.sort_values(["Seed", "Model"]).reset_index(drop=True)
            atomic_csv(raw, raw_path)
            summaries(raw, summary_path, f1_path)
            done.add(key)
            print(f"  SAVED | F1={result['F1-score']:.6f} | Acc={result['Accuracy']:.6f} | train={result['Training Time Diagnostic (s)']:.3f}s | infer={result['Inference Time Diagnostic (s)']:.3f}s")

    raw = load_raw(raw_path)
    requested = {(int(s), m) for s in args.seeds for m in args.models}
    missing = sorted(requested - pairs(raw))
    print("\n=== STATUS ===")
    print(f"Checkpoint rows: {len(raw)}")
    print(f"Requested pairs complete: {not missing}")
    print(f"Raw:      {raw_path}")
    print(f"Summary:  {summary_path}")
    print(f"F1:       {f1_path}")
    print(f"Protocol: {protocol_path}")
    if missing:
        print("Missing pairs:")
        for seed, model in missing:
            print(f"- seed={seed}, model={model}")
    else:
        print("DONE for requested model/seed scope.")


if __name__ == "__main__":
    main()
