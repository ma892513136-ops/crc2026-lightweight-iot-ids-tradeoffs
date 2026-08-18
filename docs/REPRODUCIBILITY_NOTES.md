# Reproducibility and provenance notes

## Paper-level numerical authority

`results/archived/paper_baseline_results.csv` is the frozen table of paper-level baseline values. It contains the original UNSW-NB15 baseline records and the corrected ToN_IoT resource profile used for the conference manuscript.

For ToN_IoT, the corrected memory quantity is the **peak process RSS increase observed across fit and prediction**. Wall-clock training time, inference time, throughput, and RSS memory are platform-sensitive and are not expected to match bit-for-bit on other hardware or library versions.

## Repeated-refit consistency evidence

The major-revision consistency check uses independent stratified 80:20 splits for seeds 42, 43, and 44 and independently refits RF, Logistic Regression, Decision Tree, and Naive Bayes for each seed. Mean and standard deviation are used descriptively only.

The publication-facing repeated-consistency script mirrors the executed conference protocol: ToN_IoT is one-hot encoded before the split with no explicit `float32` coercion. This detail matters for reproducing the archived seed-42 Logistic Regression result.

MLP is not included in the repeated-refit archive because the exact conference dtype path made repeated MLP fitting disproportionately expensive on the 8 GB local machine. Interrupted or diagnostic development runs are intentionally excluded from this repository.

## Archived vs generated results

- `results/archived/` contains the values used to support the paper and reviewer response.
- `results/generated/` is ignored by Git except for `.gitkeep`; new executions should write there.
- Generated resource measurements should not silently replace archived paper values.

## Scope

This package is limited to the conference study. It does not establish real IoT hardware performance, direct processor utilization, end-to-end packet latency, adversarial/evasion robustness, temporal drift robustness, or incident-response effectiveness.
