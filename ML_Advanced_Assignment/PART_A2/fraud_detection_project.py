
# PROJECT — Fraud Detection Under Extreme Imbalance
# Dataset: credit-card-style fraud dataset; uses a provided CSV if available,
# otherwise creates a reproducible synthetic fraud-style dataset with <1% positives.
#
# Deliverables:
#   output/PROJECT/
#       data_profile.csv
#       eda_class_distribution.png
#       eda_feature_separability.png
#       cv_fold_distribution.csv
#       comparison_table.csv
#       threshold_cost_table.csv
#       cost_vs_threshold.png
#       confusion_matrix_best_strategy.png
#       recommendation_report.txt
#       project_summary.csv
#       project_summary.txt
#       fraud_detection_project.ipynb
#
# Cost assumption:
#   False Negative cost = 20
#   False Positive cost = 1

import os
import time
import warnings
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
N_SPLITS = 5
FN_COST = 20
FP_COST = 1
TARGET_RECALL = 0.90

BASE_DIR = Path("output")
PROJECT_DIR = BASE_DIR / "PROJECT"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

# Put a real Kaggle/credit-card CSV here if you have one.
# Example:
# DATA_PATH = Path("data/creditcard.csv")
DATA_PATH = Path("creditcard.csv")

TARGET_CANDIDATES = ["Class", "class", "Fraud", "fraud", "is_fraud", "Is_Fraud"]

# ============================================================
# 1. LOAD DATASET OR CREATE REPRODUCIBLE FALLBACK
# ============================================================
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
        if target is None:
            raise ValueError(
                f"CSV found at {DATA_PATH}, but no target column was found. "
                f"Expected one of {TARGET_CANDIDATES}."
            )

        # Convert binary target to 0/1 where possible.
        y_raw = df[target]
        if not pd.api.types.is_numeric_dtype(y_raw):
            values = {str(v).strip().lower() for v in y_raw.dropna().unique()}
            if values <= {"0", "1"}:
                df[target] = y_raw.astype(int)
            elif values <= {"no", "yes"}:
                df[target] = y_raw.astype(str).str.lower().map({"no": 0, "yes": 1})
            else:
                raise ValueError(
                    f"Target {target!r} is not recognized as binary: {sorted(values)}"
                )

        df = df.dropna(subset=[target]).copy()
        X = df.drop(columns=[target])
        y = df[target].astype(int)

        # Keep numeric features for a compact, leakage-safe fraud workflow.
        X = X.select_dtypes(include=[np.number]).copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        # Median imputation is performed within the model pipeline later.
        source = f"Real CSV: {DATA_PATH}"
        return X, y, target, source

    # Fallback: anonymized credit-card-style synthetic data.
    X_arr, y_arr = make_classification(
        n_samples=20000,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.997, 0.003],  # 0.3% fraud
        class_sep=1.25,
        flip_y=0,
        random_state=RANDOM_STATE
    )

    columns = [f"V{i}" for i in range(1, 13)]
    X = pd.DataFrame(X_arr, columns=columns)
    y = pd.Series(y_arr, name="Class")
    source = (
        "Synthetic fallback: 20,000 anonymized credit-card-style transactions "
        "with 0.3% fraud prevalence. Replace DATA_PATH with a real dataset when available."
    )
    return X, y, "Class", source


X, y, target_name, data_source = load_data()

print("=" * 80)
print("PROJECT — FRAUD DETECTION UNDER EXTREME IMBALANCE")
print("=" * 80)
print("Data source:", data_source)
print("Shape:", X.shape)
print("Fraud count:", int(y.sum()))
print("Normal count:", int((y == 0).sum()))
print("Fraud rate:", f"{100*y.mean():.4f}%")

# ============================================================
# 2. BASIC CLEANING / DATA PROFILE
# ============================================================
# Remove constant columns and columns with no usable numeric values.
constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)

profile = pd.DataFrame({
    "feature": X.columns,
    "dtype": [str(X[c].dtype) for c in X.columns],
    "missing_count": [int(X[c].isna().sum()) for c in X.columns],
    "missing_pct": [100 * X[c].isna().mean() for c in X.columns],
    "n_unique": [int(X[c].nunique(dropna=True)) for c in X.columns]
})
profile.to_csv(PROJECT_DIR / "data_profile.csv", index=False)

# ============================================================
# 3. EDA — CLASS DISTRIBUTION
# ============================================================
counts = y.value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(["Normal", "Fraud"], [counts.get(0, 0), counts.get(1, 0)])
plt.ylabel("Number of transactions")
plt.title("Extreme Class Imbalance")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "eda_class_distribution.png", dpi=220)
plt.close()

# ============================================================
# 4. EDA — FEATURE SEPARABILITY
# ============================================================
# Compare standardized mean separation for numeric features:
# |mean_fraud - mean_normal| / pooled_std
rows = []
for c in X.columns:
    normal = X.loc[y == 0, c].dropna()
    fraud = X.loc[y == 1, c].dropna()
    if len(normal) and len(fraud):
        pooled = np.sqrt((normal.var(ddof=1) + fraud.var(ddof=1)) / 2)
        separation = abs(fraud.mean() - normal.mean()) / pooled if pooled > 0 else 0
        rows.append({
            "feature": c,
            "normal_mean": normal.mean(),
            "fraud_mean": fraud.mean(),
            "mean_separation": separation
        })

separation_df = pd.DataFrame(rows).sort_values("mean_separation", ascending=False)
separation_df.to_csv(PROJECT_DIR / "feature_separability.csv", index=False)

top_features = separation_df.head(min(6, len(separation_df)))["feature"].tolist()

fig, axes = plt.subplots(
    nrows=max(1, math.ceil(len(top_features) / 2)),
    ncols=2,
    figsize=(12, 4 * max(1, math.ceil(len(top_features) / 2)))
)
axes = np.array(axes).reshape(-1)

for ax, feature in zip(axes, top_features):
    normal = X.loc[y == 0, feature].dropna()
    fraud = X.loc[y == 1, feature].dropna()
    ax.hist(normal, bins=40, alpha=0.55, label="Normal", density=True)
    ax.hist(fraud, bins=25, alpha=0.65, label="Fraud", density=True)
    ax.set_title(feature)
    ax.legend()

for ax in axes[len(top_features):]:
    ax.axis("off")

plt.suptitle("Top Features by Class Separability", y=1.01)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "eda_feature_separability.png", dpi=220, bbox_inches="tight")
plt.close()

# ============================================================
# 5. CORRECT CROSS-VALIDATION SCHEME
# ============================================================
# The SAME folds are reused for every strategy.
# Resampling is INSIDE each pipeline, so it sees only the training fold.
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)
folds = list(skf.split(X, y))

fold_rows = []
for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
    fold_rows.append({
        "fold": fold_no,
        "train_size": len(train_idx),
        "train_fraud": int(y.iloc[train_idx].sum()),
        "train_fraud_pct": 100 * y.iloc[train_idx].mean(),
        "validation_size": len(val_idx),
        "validation_fraud": int(y.iloc[val_idx].sum()),
        "validation_fraud_pct": 100 * y.iloc[val_idx].mean()
    })
pd.DataFrame(fold_rows).to_csv(PROJECT_DIR / "cv_fold_distribution.csv", index=False)

# ============================================================
# 6. PREPROCESSING
# ============================================================
# Median imputation + standardization are inside the pipeline.
# This also prevents preprocessing leakage.
def make_logistic_pipeline(resampler=None, class_weight=None):
    # Each preprocessing step is directly inside imblearn.Pipeline.
    # This avoids nested Pipeline objects and keeps all fitting inside
    # the training fold.
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]

    if resampler is not None:
        steps.append(("resample", resampler))

    steps.append((
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=RANDOM_STATE
        )
    ))

    return Pipeline(steps)

# Four required strategies:
# 1. no correction
# 2. class_weight balanced
# 3. SMOTE
# 4. random undersampling
strategies = {
    "Baseline": make_logistic_pipeline(),
    "Class Weight Balanced": make_logistic_pipeline(
        class_weight="balanced"
    ),
    "SMOTE": make_logistic_pipeline(
        resampler=SMOTE(random_state=RANDOM_STATE)
    ),
    "Random Undersampling": make_logistic_pipeline(
        resampler=RandomUnderSampler(random_state=RANDOM_STATE)
    )
}

# ============================================================
# 7. BUSINESS COST FUNCTION
# ============================================================
def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    pr_auc = average_precision_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    total_cost = FP_COST * fp + FN_COST * fn

    return {
        "PR_AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Expected_Cost": total_cost
    }

# ============================================================
# 8. Q3-STYLE MODEL COMPARISON USING BUSINESS COST
# ============================================================
comparison_rows = []
oof_predictions = {}

for strategy_name, model in strategies.items():
    fold_metric_rows = []
    oof_prob = np.zeros(len(y))

    for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model.fit(X_train, y_train)

        prob = model.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)

        oof_prob[val_idx] = prob
        metrics = calculate_metrics(y_val, pred, prob)
        metrics["fold"] = fold_no
        fold_metric_rows.append(metrics)

    fold_df = pd.DataFrame(fold_metric_rows)

    comparison_rows.append({
        "Strategy": strategy_name,
        "Mean_PR_AUC": fold_df["PR_AUC"].mean(),
        "Std_PR_AUC": fold_df["PR_AUC"].std(),
        "Mean_Precision": fold_df["Precision"].mean(),
        "Mean_Recall": fold_df["Recall"].mean(),
        "Mean_F1": fold_df["F1"].mean(),
        "Mean_FP": fold_df["FP"].mean(),
        "Mean_FN": fold_df["FN"].mean(),
        "Mean_Expected_Cost_at_0.5": fold_df["Expected_Cost"].mean()
    })

    oof_predictions[strategy_name] = oof_prob

comparison_table = pd.DataFrame(comparison_rows)
comparison_table = comparison_table.sort_values(
    "Mean_Expected_Cost_at_0.5"
)
comparison_table.to_csv(PROJECT_DIR / "comparison_table.csv", index=False)

# ============================================================
# 9. THRESHOLD TUNING — MINIMIZE EXPECTED COST
# ============================================================
threshold_rows = []

# Use a fine threshold grid and OOF predictions.
threshold_grid = np.unique(
    np.concatenate([
        np.linspace(0.001, 0.999, 999),
        np.array([0.5])
    ])
)

for strategy_name, prob in oof_predictions.items():
    for threshold in threshold_grid:
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        cost = FP_COST * fp + FN_COST * fn

        threshold_rows.append({
            "Strategy": strategy_name,
            "Threshold": threshold,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "Expected_Cost": cost,
            "Cost_per_transaction": cost / len(y)
        })

threshold_cost_table = pd.DataFrame(threshold_rows)
threshold_cost_table.to_csv(
    PROJECT_DIR / "threshold_cost_table.csv",
    index=False
)

# Best threshold = minimum total expected cost.
best_rows = (
    threshold_cost_table
    .sort_values(
        ["Strategy", "Expected_Cost", "Threshold"],
        ascending=[True, True, False]
    )
    .groupby("Strategy", as_index=False)
    .first()
)

best_rows.to_csv(
    PROJECT_DIR / "best_threshold_by_strategy.csv",
    index=False
)

# Recommended strategy = lowest achievable expected cost.
best_overall = best_rows.sort_values(
    "Expected_Cost"
).iloc[0]

recommended_strategy = best_overall["Strategy"]
recommended_threshold = float(best_overall["Threshold"])
recommended_cost = float(best_overall["Expected_Cost"])

# ============================================================
# 10. COST VS THRESHOLD PLOT
# ============================================================
plt.figure(figsize=(10, 6))

for strategy_name in strategies:
    subset = threshold_cost_table[
        threshold_cost_table["Strategy"] == strategy_name
    ]
    plt.plot(
        subset["Threshold"],
        subset["Expected_Cost"],
        label=strategy_name
    )

plt.axvline(
    recommended_threshold,
    linestyle="--",
    label=f"Recommended threshold = {recommended_threshold:.3f}"
)
plt.xlabel("Decision threshold")
plt.ylabel("Total expected cost")
plt.title(f"Threshold Tuning: FN Cost = {FN_COST} × FP Cost")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "cost_vs_threshold.png", dpi=220)
plt.close()

# ============================================================
# 11. RECOMMENDED MODEL — OOF CONFUSION MATRIX
# ============================================================
recommended_prob = oof_predictions[recommended_strategy]
recommended_pred = (
    recommended_prob >= recommended_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y, recommended_pred, labels=[0, 1]
).ravel()

plt.figure(figsize=(6, 5))
matrix = np.array([[tn, fp], [fn, tp]])
plt.imshow(matrix, interpolation="nearest")
plt.title(
    f"Recommended Strategy: {recommended_strategy}\n"
    f"Threshold={recommended_threshold:.3f}"
)
plt.colorbar()
plt.xticks([0, 1], ["Predicted Normal", "Predicted Fraud"])
plt.yticks([0, 1], ["Actual Normal", "Actual Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "confusion_matrix_recommended.png",
    dpi=220
)
plt.close()

# ============================================================
# 12. RECOMMENDATION REPORT
# ============================================================
baseline_cost_row = best_rows[
    best_rows["Strategy"] == "Baseline"
].iloc[0]

cost_savings = (
    float(baseline_cost_row["Expected_Cost"]) - recommended_cost
)
cost_savings_pct = (
    100 * cost_savings / float(baseline_cost_row["Expected_Cost"])
    if baseline_cost_row["Expected_Cost"] > 0 else 0
)

report = f"""
PROJECT — FRAUD DETECTION UNDER EXTREME IMBALANCE
=================================================

DATASET
-------
Source:
{data_source}

Transactions:
{len(y):,}

Normal transactions:
{int((y == 0).sum()):,}

Fraud transactions:
{int((y == 1).sum()):,}

Fraud prevalence:
{100*y.mean():.4f}%


EDA FINDINGS
------------
The dataset is highly imbalanced, with fraud representing
only {100*y.mean():.4f}% of transactions.

Feature separability was examined using the standardized
difference between the fraud and normal class means.
The top features are stored in:
feature_separability.csv


CROSS-VALIDATION DESIGN
-----------------------
A {N_SPLITS}-fold StratifiedKFold was used.

The exact same folds were used for every strategy.

Any resampling operation was embedded inside the
imblearn Pipeline. Therefore SMOTE and undersampling
were fitted only on the training portion of each fold.

Imputation and scaling were also performed inside the
pipeline to prevent preprocessing leakage.


IMBALANCE STRATEGIES
--------------------
The following four strategies were compared:

1. Baseline
2. Class Weight Balanced
3. SMOTE
4. Random Undersampling

The main ranking metric was PR-AUC because fraud is rare.
Business cost was also evaluated using:

False Positive Cost = {FP_COST}
False Negative Cost = {FN_COST}

Therefore:

Expected Cost = FP × {FP_COST} + FN × {FN_COST}


FINAL COST-BASED RECOMMENDATION
--------------------------------
Recommended strategy:
{recommended_strategy}

Recommended threshold:
{recommended_threshold:.6f}

Expected total cost:
{recommended_cost:.0f}

True Negatives:
{tn}

False Positives:
{fp}

False Negatives:
{fn}

True Positives:
{tp}

Precision:
{precision_score(y, recommended_pred, zero_division=0):.6f}

Recall:
{recall_score(y, recommended_pred, zero_division=0):.6f}

F1:
{f1_score(y, recommended_pred, zero_division=0):.6f}

PR-AUC:
{average_precision_score(y, recommended_prob):.6f}


COST JUSTIFICATION
------------------
The final threshold was NOT selected using accuracy or F1.

It was selected by minimizing:

    20 × FN + 1 × FP

This is appropriate because a missed fraud transaction
is assumed to cost twenty times as much as a false alarm.

Compared with the baseline's best achievable cost under
the same threshold-search procedure:

Baseline minimum cost:
{float(baseline_cost_row["Expected_Cost"]):.0f}

Recommended minimum cost:
{recommended_cost:.0f}

Absolute cost reduction:
{cost_savings:.0f}

Percentage cost reduction:
{cost_savings_pct:.2f}%


RECOMMENDATION
--------------
The recommended strategy is {recommended_strategy} with a
decision threshold of {recommended_threshold:.4f}.

The recommendation is based on expected business cost,
not simply the highest PR-AUC. This matters because a
fraud system should explicitly reflect the much larger
business consequence of false negatives.

Operationally, the selected probability threshold should
be monitored after deployment because fraud prevalence,
fraud patterns, and the cost assumptions can change over
time.


KEY OUTPUT FILES
----------------
comparison_table.csv
best_threshold_by_strategy.csv
threshold_cost_table.csv
eda_class_distribution.png
eda_feature_separability.png
cost_vs_threshold.png
confusion_matrix_recommended.png
project_summary.csv
project_summary.txt
"""

with open(
    PROJECT_DIR / "recommendation_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

# ============================================================
# 13. PROJECT SUMMARY CSV
# ============================================================
summary = pd.DataFrame([{
    "dataset_source": data_source,
    "n_transactions": len(y),
    "n_features": X.shape[1],
    "fraud_count": int(y.sum()),
    "fraud_rate_pct": 100 * y.mean(),
    "cv_folds": N_SPLITS,
    "fn_cost": FN_COST,
    "fp_cost": FP_COST,
    "recommended_strategy": recommended_strategy,
    "recommended_threshold": recommended_threshold,
    "recommended_pr_auc": average_precision_score(y, recommended_prob),
    "recommended_precision": precision_score(y, recommended_pred, zero_division=0),
    "recommended_recall": recall_score(y, recommended_pred, zero_division=0),
    "recommended_f1": f1_score(y, recommended_pred, zero_division=0),
    "recommended_expected_cost": recommended_cost,
    "recommended_TN": tn,
    "recommended_FP": fp,
    "recommended_FN": fn,
    "recommended_TP": tp,
    "baseline_best_cost": float(baseline_cost_row["Expected_Cost"]),
    "cost_reduction": cost_savings,
    "cost_reduction_pct": cost_savings_pct
}])

summary.to_csv(
    PROJECT_DIR / "project_summary.csv",
    index=False
)

with open(
    PROJECT_DIR / "project_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("\n" + "=" * 80)
print("PROJECT COMPLETED")
print("=" * 80)
print("Recommended strategy:", recommended_strategy)
print("Recommended threshold:", round(recommended_threshold, 6))
print("Expected cost:", round(recommended_cost, 2))
print("PR-AUC:", round(average_precision_score(y, recommended_prob), 6))
print("Precision:", round(precision_score(y, recommended_pred, zero_division=0), 6))
print("Recall:", round(recall_score(y, recommended_pred, zero_division=0), 6))
print("\nAll project outputs saved to:")
print(PROJECT_DIR.resolve())
