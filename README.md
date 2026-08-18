# CRC 2026 Lightweight IoT IDS Trade-off Reproducibility

This repository contains a compact reproducibility package for the paper:

**Analysis of Detection and Resource Trade-offs in Lightweight IoT Intrusion Detection Models**

It is intentionally limited to the conference paper; unrelated experimental workflows are omitted.

## Repository contents

- `src/run_unsw_nb15_baseline.py` — UNSW-NB15 full-feature baseline runner for RF, SVM, MLP, and Logistic Regression using the official predefined train/test split.
- `src/run_ton_iot_baseline.py` — ToN_IoT full-feature baseline runner for RF, Logistic Regression, MLP, Decision Tree, and Naive Bayes.
- `src/run_ton_iot_repeated_consistency.py` — three-seed repeated-refit consistency check used in the major revision. The default scope is RF, Logistic Regression, Decision Tree, and Naive Bayes with seeds 42, 43, and 44.
- `src/verify_archived_results.py` — verifies the archived repeated-refit summaries and the seed-42 detection values against the archived paper baseline.
- `results/archived/paper_baseline_results.csv` — numerical values used as the paper-level baseline authority.
- `results/archived/repeated_consistency/` — completed repeated-refit outputs used in the major revision.
- `docs/REPRODUCIBILITY_NOTES.md` — protocol boundaries and provenance notes.

## Data

Raw datasets are **not redistributed** in this repository. Place the original dataset files at:

```text
data/raw/unsw_nb15/UNSW_NB15_training-set.csv
data/raw/unsw_nb15/UNSW_NB15_testing-set.csv
data/raw/ton_iot/train_test_network.csv
```

See `data/README.md` for preprocessing expectations.

## Environment

The repeated-refit consistency check was executed on macOS 14.5 (arm64), Python 3.13.3, NumPy 2.4.4, pandas 3.0.2, and scikit-learn 1.8.0. The hardware environment was a MacBook Air with Apple M3 and 8 GB RAM, without a discrete GPU.

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the archived consistency check

From the repository root:

```bash
python src/run_ton_iot_repeated_consistency.py   --project_root .   --seeds 42 43 44
```

The default repeated-refit scope intentionally excludes MLP because the exact conference preprocessing/dtype path made the repeated MLP run disproportionately expensive in the constrained local environment. MLP remains part of the single-run ToN_IoT baseline. The consistency analysis is descriptive and does not claim inferential statistical significance.

## Run the baseline scripts

UNSW-NB15:

```bash
python src/run_unsw_nb15_baseline.py --project_root .
```

ToN_IoT:

```bash
python src/run_ton_iot_baseline.py --project_root .
```

Resource measurements (training time, inference time, throughput, and RSS-based memory) are platform-dependent. Re-running the scripts on a different machine or software stack should reproduce the evaluation logic and detection metrics more closely than the wall-clock resource values.

## Verify the archived results

```bash
python src/verify_archived_results.py
```

The verifier checks that the three-seed summary is internally consistent and that seed 42 reproduces the archived ToN_IoT detection values for RF, Logistic Regression, Decision Tree, and Naive Bayes to the precision stored in the paper baseline.

## Scope boundary

This repository supports a controlled offline detection–resource trade-off analysis. It does not claim real-device deployment, end-to-end packet latency, direct CPU-utilization profiling, adversarial robustness, concept-drift robustness, or operational incident-response validation.
