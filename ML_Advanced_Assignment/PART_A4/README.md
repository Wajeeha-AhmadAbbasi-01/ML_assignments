# PART A4 — The Cost of Skipping Scaling

## Overview

This project demonstrates what happens to different machine learning models when you **skip feature scaling**. Using the Breast Cancer Wisconsin dataset, we deliberately create mixed numerical scales (multiplying every third feature by 1000) and compare each model's performance **with** and **without** `StandardScaler`.

The goal is to show that scaling matters *a lot* for some models and *barely at all* for others — and to explain **why** mechanistically.

---

## Quick Start

```bash
pip install numpy pandas matplotlib scikit-learn
python cost_of_skipping_scaling.py
```

All outputs (CSV tables, PNG charts, and a text summary) are saved in `./outputs`.

---

## Dataset

- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569
- **Features:** 30 numeric features
- **Target:** Binary (malignant / benign)
- **Scale manipulation:** Every third feature (`X.iloc[:, ::3]`) is multiplied by **1000** to simulate a realistic scenario where features live on wildly different scales.

**Split:** Stratified 80/20 train/test split with `random_state=42` (reproducible).

---

## Models Compared

Four models are each trained **twice** — once on raw features, once inside a `Pipeline` with `StandardScaler`:

| Model | Key Settings | Scaling Sensitive? |
|---|---|---|
| **KNN** | `n_neighbors=5` | ✅ Yes |
| **RBF SVM** | `kernel="rbf"`, `C=1.0`, `gamma="scale"` | ✅ Yes |
| **Regularized Logistic Regression** | `C=1.0`, `max_iter=5000` | ✅ Yes (L2 penalty + optimization) |
| **Random Forest** | `n_estimators=200`, `random_state=42` | ❌ Mostly no |

> **Best practice demonstrated:** The scaler is always placed *inside* a `Pipeline`, so it is fitted only on the training split — no data leakage.

---

## Metrics Collected

For every model/version pair:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 score**
- **Convergence warning** (captured via `warnings.catch_warnings`)

Then the script computes:

- `Accuracy change = scaled − raw`
- `F1 change = scaled − raw`

These deltas are bucketed into practical labels:

| Label | Rule |
|---|---|
| **Substantial change** | \|Δaccuracy\| ≥ 0.02 **or** \|ΔF1\| ≥ 0.02 |
| **Little/no measurable change** | \|Δaccuracy\| ≤ 0.005 **and** \|ΔF1\| ≤ 0.005 |
| **Small/moderate change** | Anything in between |

---

## Why Scaling Matters (or Doesn't) — Per Model

### KNN
> KNN uses distances. Large-valued features can dominate Euclidean distance, changing which rows count as nearest neighbors.

A feature on the scale of `~1000` will overwhelm a feature on the scale of `~1` when computing distance — the model effectively ignores the small-scale features.

### RBF SVM
> The RBF kernel depends on squared distances: `exp(-gamma * ||x-z||²)`. Large-scale features can dominate similarities and change the decision boundary.

Because the kernel is exponential in squared distance, the *largest* feature essentially decides similarity. Unscaled data can collapse the kernel's useful range.

### Regularized Logistic Regression
> L2 regularization penalizes coefficient squares. Feature units affect coefficient magnitudes, so the same penalty treats differently scaled features unevenly. Scaling can also improve optimization.

Two effects compound here:
1. **Regularization is scale-dependent** — a feature multiplied by 1000 needs a coefficient 1000× smaller, so the L2 penalty hits it differently.
2. **Optimization is slower/noisier** on unscaled features (hence the `ConvergenceWarning` capture).

### Random Forest
> Trees split one feature at a time using thresholds. Positive rescaling preserves feature ordering, so equivalent splits remain possible. Tree ensembles usually do not need scaling.

This is the key insight: **trees are invariant to monotonic transformations**. Multiplying a feature by 1000 doesn't change which thresholds best separate the classes. Expect ≈zero difference.

---

## Outputs

### Tables (CSV)

| File | Contents |
|---|---|
| `results_comparison_table.csv` | Wide table: every metric × model × version |
| `model_metrics_long.csv` | Long format, one row per model/version |
| `scaling_effect_and_explanations.csv` | Δaccuracy, ΔF1, effect label, and mechanistic explanation per model |

### Charts (PNG)

| File | Contents |
|---|---|
| `accuracy_comparison.png` | Grouped bar chart: raw vs. scaled accuracy per model |
| `f1_comparison.png` | Grouped bar chart: raw vs. scaled F1 per model |

### Summary

| File | Contents |
|---|---|
| `SUMMARY.txt` | Human-readable report: dataset description, results table, effect table, per-model explanations, and the reporting rule |

---

## Expected Takeaways

1. **Distance-based models (KNN, SVM)** — scaling is *essential*. Expect large performance swings.
2. **Regularized linear models (Logistic Regression)** — scaling changes the effective regularization and often fixes convergence warnings.
3. **Tree ensembles (Random Forest)** — scaling is *unnecessary*. Expect negligible change and no convergence issues.
4. **Always use a `Pipeline`** — it guarantees the scaler is fit on training data only, preventing leakage during CV or test evaluation.

> ⚠️ **Honest reporting:** The script measures actual results rather than assuming scaling always helps. The labels ("substantial", "little/no measurable change") are **practical cutoffs**, not universal statistical rules. Interpret the printed values in context — dataset, split, and hyperparameters all affect outcomes.

---

## Project Structure

```
.
├── cost_of_skipping_scaling.py       # Main experiment script
├── README.md                         # This file
└── outputs/
    ├── results_comparison_table.csv
    ├── model_metrics_long.csv
    ├── scaling_effect_and_explanations.csv
    ├── accuracy_comparison.png
    ├── f1_comparison.png
    └── SUMMARY.txt
```

---

## Notes on Reproducibility

- All randomness is seeded (`random_state=42` for the split, Logistic Regression, and Random Forest).
- The scale manipulation (`X.iloc[:, ::3] *= 1000`) is deterministic.
- Running the script twice produces identical outputs.

---

## Related Work

This file is **Part A4** of a broader feature-scaling study. The companion script `feature_scaling_experiments_with_outputs.py` covers Questions 1–5:

| Question | Focus |
|---|---|
| Q1 | KNN accuracy: scaled vs. unscaled |
| Q2 | PCA explained-variance with/without scaling |
| Q3 | `ColumnTransformer` with three scalers + cross-validation |
| Q4 | Ridge coefficient stability across bootstrap samples |
| Q5 | Logistic Regression (SAGA) convergence iterations |

Together the two scripts give a full tour of *when* scaling matters, *how much*, and *why*.