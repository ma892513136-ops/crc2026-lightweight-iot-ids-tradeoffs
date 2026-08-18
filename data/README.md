# Data placement and preprocessing

Raw datasets are not included. Use the original UNSW-NB15 and ToN_IoT distributions and place the files at the paths shown below.

```text
data/raw/unsw_nb15/UNSW_NB15_training-set.csv
data/raw/unsw_nb15/UNSW_NB15_testing-set.csv
data/raw/ton_iot/train_test_network.csv
```

## UNSW-NB15

- Binary target: `label`.
- Drop `id` and `attack_cat` from the predictors.
- One-hot encode categorical variables independently in the predefined training and testing files, then align the test columns to the training feature space.
- The paper uses the official predefined train/test split.

## ToN_IoT network subset

- Binary target: `label`.
- Drop `type`, `src_ip`, and `dst_ip` from the predictors.
- For conference-protocol consistency, apply `pandas.get_dummies` to the full predictor matrix before the stratified 80:20 split.
- Baseline split seed: 42.
- Repeated-refit consistency seeds: 42, 43, 44.
- The resulting conference feature space contained 973 encoded features in the archived run.

The repository preserves the conference protocol as executed; it does not reinterpret it as a general best-practice preprocessing recommendation.
