# ================================================================
# IMBALANCED CLASSIFICATION ASSIGNMENT
#
# Q1 - SMOTE inside StratifiedKFold + Pipeline
# Q2 - Compare four imbalance strategies
# Q3 - Threshold tuning for target recall
# Q4 - SMOTE implementation from scratch
# Q5 - Balanced Random Forest vs Random Forest + SMOTE
#
# All results are automatically saved inside:
# output/
#
# ================================================================


# ================================================================
# 1. IMPORTS
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier


# ================================================================
# 2. CONFIGURATION
# ================================================================

RANDOM_STATE = 42

N_SAMPLES = 10000
N_FEATURES = 10

N_SPLITS = 5

TARGET_RECALL = 0.90

OUTPUT_DIR = "output"

Q1_DIR = os.path.join(OUTPUT_DIR, "Q1")
Q2_DIR = os.path.join(OUTPUT_DIR, "Q2")
Q3_DIR = os.path.join(OUTPUT_DIR, "Q3")
Q4_DIR = os.path.join(OUTPUT_DIR, "Q4")
Q5_DIR = os.path.join(OUTPUT_DIR, "Q5")


# ================================================================
# 3. CREATE OUTPUT DIRECTORIES
# ================================================================

for directory in [
    OUTPUT_DIR,
    Q1_DIR,
    Q2_DIR,
    Q3_DIR,
    Q4_DIR,
    Q5_DIR
]:
    os.makedirs(directory, exist_ok=True)


# ================================================================
# 4. CREATE FRAUD-STYLE DATASET
# ================================================================

print("\n" + "=" * 80)
print("CREATING FRAUD-STYLE DATASET")
print("=" * 80)

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=1,

    # 99% normal / 1% fraud
    weights=[0.99, 0.01],

    flip_y=0,
    class_sep=1.0,
    random_state=RANDOM_STATE
)


print("\nDataset shape:")
print(X.shape)

class_counts = pd.Series(y).value_counts().sort_index()

print("\nClass distribution:")
print(class_counts)

print("\nClass percentages:")
print(
    pd.Series(y)
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)


# Save basic dataset information
dataset_summary = pd.DataFrame({
    "Class": ["Normal", "Fraud"],
    "Label": [0, 1],
    "Count": [
        np.sum(y == 0),
        np.sum(y == 1)
    ],
    "Percentage": [
        np.mean(y == 0) * 100,
        np.mean(y == 1) * 100
    ]
})

dataset_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_class_distribution.csv"
    ),
    index=False
)


# ================================================================
# 5. CREATE ONE COMMON STRATIFIED K-FOLD SPLIT
# ================================================================
#
# IMPORTANT:
#
# We create the folds ONCE and reuse them for Q1, Q2 and Q5.
#
# This makes comparisons fair.
#
# ================================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

folds = list(
    skf.split(X, y)
)

print("\nNumber of CV folds:", len(folds))


# ================================================================
# Q1
# SMOTE INSIDE IMBLEARN PIPELINE
# ================================================================

print("\n" + "=" * 80)
print("Q1 - SMOTE + STRATIFIED K-FOLD")
print("=" * 80)

q1_results = []


for fold_number, (train_idx, val_idx) in enumerate(
    folds,
    start=1
):

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]


    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # SMOTE is INSIDE the pipeline.
    #
    # Therefore, SMOTE is fitted only on the training fold.
    #
    # ------------------------------------------------------------

    q1_pipeline = Pipeline([
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])


    # Train
    q1_pipeline.fit(
        X_train,
        y_train
    )


    # Validation probabilities
    y_prob = q1_pipeline.predict_proba(
        X_val
    )[:, 1]


    # PR-AUC
    pr_auc = average_precision_score(
        y_val,
        y_prob
    )


    q1_results.append({
        "Fold": fold_number,
        "PR_AUC": pr_auc,
        "Validation_Size": len(y_val),
        "Validation_Fraud_Count": int(np.sum(y_val == 1)),
        "Validation_Normal_Count": int(np.sum(y_val == 0))
    })


    print(
        f"Fold {fold_number}: "
        f"PR-AUC = {pr_auc:.4f}"
    )


# Convert to DataFrame
q1_results_df = pd.DataFrame(
    q1_results
)


# Save fold results
q1_results_df.to_csv(
    os.path.join(
        Q1_DIR,
        "q1_fold_results.csv"
    ),
    index=False
)


q1_mean = q1_results_df["PR_AUC"].mean()
q1_std = q1_results_df["PR_AUC"].std()


q1_summary = f"""
Q1 - SMOTE INSIDE STRATIFIED K-FOLD

Method:
SMOTE + Logistic Regression

Cross-validation:
{N_SPLITS}-fold StratifiedKFold

Important:
SMOTE was placed inside an imblearn Pipeline.
Therefore, SMOTE was fitted only on each training fold.

Mean PR-AUC:
{q1_mean:.6f}

Standard Deviation:
{q1_std:.6f}

Interpretation:
This evaluates whether SMOTE improves detection of the
rare fraud class while avoiding data leakage.
"""

with open(
    os.path.join(
        Q1_DIR,
        "q1_summary.txt"
    ),
    "w"
) as file:

    file.write(
        q1_summary
    )


print("\nQ1 Mean PR-AUC:", round(q1_mean, 4))
print("Q1 Std PR-AUC:", round(q1_std, 4))


# ================================================================
# Q2
# COMPARE FOUR IMBALANCE STRATEGIES
# ================================================================

print("\n" + "=" * 80)
print("Q2 - COMPARISON OF IMBALANCE STRATEGIES")
print("=" * 80)


models_q2 = {

    "Baseline": Pipeline([
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]),


    "Class Weight": Pipeline([
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]),


    "SMOTE": Pipeline([
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]),


    "Random Undersampling": Pipeline([
        (
            "undersampler",
            RandomUnderSampler(
                random_state=RANDOM_STATE
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])
}


q2_results = []


for model_name, model in models_q2.items():

    print(
        f"\nRunning: {model_name}"
    )

    fold_scores = []


    for fold_number, (train_idx, val_idx) in enumerate(
        folds,
        start=1
    ):

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]


        model.fit(
            X_train,
            y_train
        )


        y_prob = model.predict_proba(
            X_val
        )[:, 1]


        pr_auc = average_precision_score(
            y_val,
            y_prob
        )


        fold_scores.append(
            pr_auc
        )


    q2_results.append({
        "Strategy": model_name,
        "Mean_PR_AUC": np.mean(fold_scores),
        "Std_PR_AUC": np.std(fold_scores),

        "Fold_1_PR_AUC": fold_scores[0],
        "Fold_2_PR_AUC": fold_scores[1],
        "Fold_3_PR_AUC": fold_scores[2],
        "Fold_4_PR_AUC": fold_scores[3],
        "Fold_5_PR_AUC": fold_scores[4]
    })


# Create comparison table
q2_table = pd.DataFrame(
    q2_results
)


# Sort best to worst
q2_table = q2_table.sort_values(
    "Mean_PR_AUC",
    ascending=False
)


# Save
q2_table.to_csv(
    os.path.join(
        Q2_DIR,
        "q2_comparison.csv"
    ),
    index=False
)


best_q2_strategy = q2_table.iloc[0]["Strategy"]
best_q2_score = q2_table.iloc[0]["Mean_PR_AUC"]


print("\nQ2 Comparison:")
print(
    q2_table[
        [
            "Strategy",
            "Mean_PR_AUC",
            "Std_PR_AUC"
        ]
    ].to_string(index=False)
)


q2_summary = f"""
Q2 - COMPARISON OF IMBALANCE STRATEGIES

Strategies compared:

1. Baseline
   No imbalance correction.

2. Class Weight
   Uses class_weight='balanced'.

3. SMOTE
   Creates synthetic minority observations.

4. Random Undersampling
   Removes majority-class observations.

All strategies used:
{N_SPLITS}-fold StratifiedKFold

The EXACT SAME CV folds were used for all methods.

Best strategy according to mean PR-AUC:
{best_q2_strategy}

Best mean PR-AUC:
{best_q2_score:.6f}

Interpretation:
This comparison determines which imbalance-handling
strategy provides the strongest fraud-detection performance
for the same classifier and same validation folds.
"""

with open(
    os.path.join(
        Q2_DIR,
        "q2_summary.txt"
    ),
    "w"
) as file:

    file.write(
        q2_summary
    )


# ================================================================
# Q3
# THRESHOLD TUNING FOR 90% RECALL
# ================================================================

print("\n" + "=" * 80)
print("Q3 - THRESHOLD TUNING")
print("=" * 80)


# ------------------------------------------------------------
# Generate out-of-fold predictions
# ------------------------------------------------------------
#
# This gives every observation a probability from a model
# that did NOT train on that observation.
#
# ------------------------------------------------------------

oof_prob = np.zeros(
    len(y)
)


q3_model = Pipeline([
    (
        "smote",
        SMOTE(
            random_state=RANDOM_STATE
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )
])


for fold_number, (train_idx, val_idx) in enumerate(
    folds,
    start=1
):

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]


    q3_model.fit(
        X_train,
        y_train
    )


    oof_prob[val_idx] = q3_model.predict_proba(
        X_val
    )[:, 1]


# ------------------------------------------------------------
# Precision-recall curve
# ------------------------------------------------------------

precision, recall, thresholds = precision_recall_curve(
    y,
    oof_prob
)


# Important:
#
# precision and recall contain one extra value compared
# with thresholds.
#
# Therefore use:
#
# precision[:-1]
# recall[:-1]
#
# when pairing them with thresholds.
# ------------------------------------------------------------

q3_threshold_table = pd.DataFrame({
    "Threshold": thresholds,
    "Precision": precision[:-1],
    "Recall": recall[:-1]
})


# ------------------------------------------------------------
# Find thresholds where recall >= 90%
# ------------------------------------------------------------

eligible = q3_threshold_table[
    q3_threshold_table["Recall"] >= TARGET_RECALL
].copy()


if len(eligible) == 0:

    print(
        "\nWARNING:"
        " No threshold achieved the target recall."
    )

    best_threshold = 0.0

    final_predictions = (
        oof_prob >= best_threshold
    ).astype(int)

else:

    # Calculate false positives for every candidate threshold

    false_positives = []


    for threshold in eligible["Threshold"]:

        predictions = (
            oof_prob >= threshold
        ).astype(int)


        cm_temp = confusion_matrix(
            y,
            predictions
        )


        tn_temp, fp_temp, fn_temp, tp_temp = (
            cm_temp.ravel()
        )


        false_positives.append(
            fp_temp
        )


    eligible["False_Positives"] = (
        false_positives
    )


    # --------------------------------------------------------
    # Select threshold:
    #
    # 1. Recall must be >= 90%
    # 2. Among those, minimize false positives
    # 3. If tied, prefer higher threshold
    # --------------------------------------------------------

    eligible = eligible.sort_values(
        [
            "False_Positives",
            "Threshold"
        ],
        ascending=[
            True,
            False
        ]
    )


    best = eligible.iloc[0]

    best_threshold = best["Threshold"]


    final_predictions = (
        oof_prob >= best_threshold
    ).astype(int)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y,
    final_predictions
)


tn, fp, fn, tp = cm.ravel()


final_precision = precision_score(
    y,
    final_predictions,
    zero_division=0
)


final_recall = recall_score(
    y,
    final_predictions,
    zero_division=0
)


final_f1 = f1_score(
    y,
    final_predictions,
    zero_division=0
)


# ------------------------------------------------------------
# Save confusion matrix
# ------------------------------------------------------------

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Normal",
        "Actual Fraud"
    ],
    columns=[
        "Predicted Normal",
        "Predicted Fraud"
    ]
)


cm_df.to_csv(
    os.path.join(
        Q3_DIR,
        "q3_confusion_matrix.csv"
    )
)


# Save threshold table
q3_threshold_table.to_csv(
    os.path.join(
        Q3_DIR,
        "q3_threshold_results.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Precision-recall curve plot
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall,
    precision,
    label="SMOTE + Logistic Regression"
)

plt.axvline(
    TARGET_RECALL,
    linestyle="--",
    label="Target Recall = 90%"
)

plt.scatter(
    final_recall,
    final_precision,
    s=100,
    label=f"Selected threshold = {best_threshold:.4f}"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve with Threshold Selection"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        Q3_DIR,
        "q3_precision_recall_curve.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Q3 summary
# ------------------------------------------------------------

q3_summary = f"""
Q3 - THRESHOLD TUNING

Target recall:
{TARGET_RECALL:.2%}

Selected threshold:
{best_threshold:.6f}

Resulting precision:
{final_precision:.6f}

Resulting recall:
{final_recall:.6f}

Resulting F1:
{final_f1:.6f}

Confusion Matrix:

True Negatives:
{tn}

False Positives:
{fp}

False Negatives:
{fn}

True Positives:
{tp}

Interpretation:
The default classification threshold of 0.50 is not
necessarily optimal for fraud detection.

The threshold was selected so that recall reaches at
least the target of 90%, while minimizing false positives
among eligible thresholds.

This represents a business trade-off between catching
fraud and generating unnecessary fraud alerts.
"""

with open(
    os.path.join(
        Q3_DIR,
        "q3_summary.txt"
    ),
    "w"
) as file:

    file.write(
        q3_summary
    )


print("\nSelected threshold:", round(best_threshold, 6))
print("Precision:", round(final_precision, 4))
print("Recall:", round(final_recall, 4))
print("F1:", round(final_f1, 4))

print("\nConfusion Matrix:")
print(cm)


# ================================================================
# Q4
# SMOTE FROM SCRATCH
# ================================================================

print("\n" + "=" * 80)
print("Q4 - SMOTE FROM SCRATCH")
print("=" * 80)


# ------------------------------------------------------------
# Create 2D toy dataset
# ------------------------------------------------------------

rng = np.random.RandomState(
    RANDOM_STATE
)


# Majority class
majority = rng.normal(
    loc=[0, 0],
    scale=1.0,
    size=(100, 2)
)


# Minority class
minority = rng.normal(
    loc=[3, 3],
    scale=0.5,
    size=(10, 2)
)


# ------------------------------------------------------------
# SMOTE algorithm from scratch
# ------------------------------------------------------------

def smote_from_scratch(
    X_minority,
    n_synthetic,
    k_neighbors=5,
    random_state=42
):

    rng = np.random.RandomState(
        random_state
    )


    # Find nearest minority neighbors
    nn = NearestNeighbors(
        n_neighbors=k_neighbors + 1
    )


    nn.fit(
        X_minority
    )


    distances, indices = nn.kneighbors(
        X_minority
    )


    synthetic_samples = []


    for _ in range(
        n_synthetic
    ):

        # ----------------------------------------------------
        # Step 1:
        # Randomly select a minority observation
        # ----------------------------------------------------

        i = rng.randint(
            0,
            len(X_minority)
        )


        # ----------------------------------------------------
        # Step 2:
        # Get its minority neighbors
        # ----------------------------------------------------

        neighbors = indices[i][1:]


        # ----------------------------------------------------
        # Step 3:
        # Randomly choose one neighbor
        # ----------------------------------------------------

        neighbor_index = rng.choice(
            neighbors
        )


        x_i = X_minority[i]

        x_neighbor = X_minority[
            neighbor_index
        ]


        # ----------------------------------------------------
        # Step 4:
        # Generate random interpolation value
        # ----------------------------------------------------

        alpha = rng.random()


        # ----------------------------------------------------
        # Step 5:
        # Create synthetic observation
        # ----------------------------------------------------

        synthetic = (
            x_i
            +
            alpha *
            (
                x_neighbor
                -
                x_i
            )
        )


        synthetic_samples.append(
            synthetic
        )


    return np.array(
        synthetic_samples
    )


# ------------------------------------------------------------
# Generate 90 synthetic minority observations
# ------------------------------------------------------------

synthetic_minority = smote_from_scratch(
    X_minority=minority,
    n_synthetic=90,
    k_neighbors=5,
    random_state=RANDOM_STATE
)


print(
    "Original minority samples:",
    len(minority)
)

print(
    "Synthetic minority samples:",
    len(synthetic_minority)
)


# ------------------------------------------------------------
# Save original minority observations
# ------------------------------------------------------------

original_minority_df = pd.DataFrame(
    minority,
    columns=[
        "Feature_1",
        "Feature_2"
    ]
)


original_minority_df.to_csv(
    os.path.join(
        Q4_DIR,
        "q4_original_minority.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Save synthetic observations
# ------------------------------------------------------------

synthetic_minority_df = pd.DataFrame(
    synthetic_minority,
    columns=[
        "Feature_1",
        "Feature_2"
    ]
)


synthetic_minority_df.to_csv(
    os.path.join(
        Q4_DIR,
        "q4_synthetic_minority.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Q4 visualization
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    majority[:, 0],
    majority[:, 1],
    alpha=0.4,
    label="Majority"
)


plt.scatter(
    minority[:, 0],
    minority[:, 1],
    s=80,
    label="Original Minority"
)


plt.scatter(
    synthetic_minority[:, 0],
    synthetic_minority[:, 1],
    marker="x",
    s=50,
    label="Synthetic Minority"
)


plt.xlabel(
    "Feature 1"
)

plt.ylabel(
    "Feature 2"
)

plt.title(
    "SMOTE From Scratch: Original vs Synthetic Points"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        Q4_DIR,
        "q4_smote_visualization.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Q4 summary
# ------------------------------------------------------------

q4_summary = f"""
Q4 - SMOTE FROM SCRATCH

Original minority observations:
{len(minority)}

Synthetic minority observations:
{len(synthetic_minority)}

Algorithm:

1. Select a minority observation.
2. Find its nearest minority neighbors.
3. Select one neighbor randomly.
4. Generate a random interpolation value.
5. Create a synthetic observation between
   the selected observation and its neighbor.

Core idea:

Synthetic =
Original +
random_value *
(Neighbor - Original)

No imbalanced-learn SMOTE implementation
was used for this question.

The visualization shows the original minority
points and the synthetic points created between
minority neighbors.
"""

with open(
    os.path.join(
        Q4_DIR,
        "q4_summary.txt"
    ),
    "w"
) as file:

    file.write(
        q4_summary
    )


# ================================================================
# Q5
# BALANCED RANDOM FOREST VS RANDOM FOREST + SMOTE
# ================================================================

print("\n" + "=" * 80)
print("Q5 - BALANCED RANDOM FOREST VS RANDOM FOREST + SMOTE")
print("=" * 80)


# ------------------------------------------------------------
# Random Forest + SMOTE
# ------------------------------------------------------------

rf_smote = Pipeline([
    (
        "smote",
        SMOTE(
            random_state=RANDOM_STATE
        )
    ),

    (
        "rf",
        RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])


# ------------------------------------------------------------
# Balanced Random Forest
# ------------------------------------------------------------

balanced_rf = BalancedRandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


models_q5 = {

    "Random Forest + SMOTE":
        rf_smote,

    "Balanced Random Forest":
        balanced_rf
}


q5_results = []


for model_name, model in models_q5.items():

    print(
        f"\nRunning: {model_name}"
    )


    fold_pr_auc = []
    fold_precision = []
    fold_recall = []
    fold_f1 = []


    for fold_number, (train_idx, val_idx) in enumerate(
        folds,
        start=1
    ):

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]


        # Train
        model.fit(
            X_train,
            y_train
        )


        # Probability
        y_prob = model.predict_proba(
            X_val
        )[:, 1]


        # Default threshold = 0.50
        y_pred = (
            y_prob >= 0.50
        ).astype(int)


        # Metrics
        pr_auc = average_precision_score(
            y_val,
            y_prob
        )


        precision_value = precision_score(
            y_val,
            y_pred,
            zero_division=0
        )


        recall_value = recall_score(
            y_val,
            y_pred,
            zero_division=0
        )


        f1_value = f1_score(
            y_val,
            y_pred,
            zero_division=0
        )


        fold_pr_auc.append(
            pr_auc
        )

        fold_precision.append(
            precision_value
        )

        fold_recall.append(
            recall_value
        )

        fold_f1.append(
            f1_value
        )


        print(
            f"  Fold {fold_number}: "
            f"PR-AUC={pr_auc:.4f}, "
            f"Precision={precision_value:.4f}, "
            f"Recall={recall_value:.4f}, "
            f"F1={f1_value:.4f}"
        )


    # Average CV metrics
    q5_results.append({

        "Model": model_name,

        "Mean_PR_AUC":
            np.mean(fold_pr_auc),

        "Std_PR_AUC":
            np.std(fold_pr_auc),

        "Mean_Precision":
            np.mean(fold_precision),

        "Mean_Recall":
            np.mean(fold_recall),

        "Mean_F1":
            np.mean(fold_f1)
    })


# ------------------------------------------------------------
# Q5 comparison table
# ------------------------------------------------------------

q5_table = pd.DataFrame(
    q5_results
)


q5_table = q5_table.sort_values(
    "Mean_PR_AUC",
    ascending=False
)


q5_table.to_csv(
    os.path.join(
        Q5_DIR,
        "q5_comparison.csv"
    ),
    index=False
)


best_q5_model = q5_table.iloc[0]["Model"]
best_q5_score = q5_table.iloc[0]["Mean_PR_AUC"]


print("\nQ5 Comparison:")
print(
    q5_table.to_string(
        index=False
    )
)


q5_summary = f"""
Q5 - BALANCED RANDOM FOREST VS RANDOM FOREST + SMOTE

Models:

1. Random Forest + SMOTE
   SMOTE is applied to each training fold before
   fitting the Random Forest.

2. Balanced Random Forest
   Uses balanced sampling internally while
   constructing the forest.

The SAME StratifiedKFold splits were used
for both models.

Best model according to mean PR-AUC:
{best_q5_model}

Best mean PR-AUC:
{best_q5_score:.6f}

Interpretation:
This comparison evaluates two different ways
of combining ensemble learning with imbalance handling.
"""

with open(
    os.path.join(
        Q5_DIR,
        "q5_summary.txt"
    ),
    "w"
) as file:

    file.write(
        q5_summary
    )


# ================================================================
# FINAL SUMMARY TABLE
# ================================================================

print("\n" + "=" * 80)
print("CREATING FINAL SUMMARY")
print("=" * 80)


final_summary = pd.DataFrame({

    "Question": [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5"
    ],

    "Purpose": [
        "Evaluate SMOTE with leakage-free cross-validation",
        "Compare four imbalance strategies",
        "Tune threshold to achieve target recall",
        "Understand SMOTE interpolation from scratch",
        "Compare Balanced RF with RF + SMOTE"
    ],

    "Main_Result": [
        f"Mean PR-AUC = {q1_mean:.6f}",
        f"Best Strategy = {best_q2_strategy}",
        f"Threshold = {best_threshold:.6f}",
        f"{len(synthetic_minority)} synthetic samples created",
        f"Best Model = {best_q5_model}"
    ]
})


final_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_summary.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Human-readable final summary
# ------------------------------------------------------------

final_text = f"""
============================================================
IMBALANCED CLASSIFICATION ASSIGNMENT - FINAL SUMMARY
============================================================

DATASET
------------------------------------------------------------
Samples: {N_SAMPLES}
Features: {N_FEATURES}
Normal cases: {np.sum(y == 0)}
Fraud cases: {np.sum(y == 1)}
Fraud rate: {np.mean(y == 1) * 100:.2f}%


============================================================
Q1 - SMOTE + STRATIFIED K-FOLD
============================================================

Mean PR-AUC:
{q1_mean:.6f}

Std PR-AUC:
{q1_std:.6f}

Result:
SMOTE was applied inside the Pipeline, so it was fitted
only on each training fold.


============================================================
Q2 - FOUR IMBALANCE STRATEGIES
============================================================

Best strategy:
{best_q2_strategy}

Best mean PR-AUC:
{best_q2_score:.6f}


============================================================
Q3 - THRESHOLD TUNING
============================================================

Target recall:
{TARGET_RECALL:.2%}

Selected threshold:
{best_threshold:.6f}

Precision:
{final_precision:.6f}

Recall:
{final_recall:.6f}

F1:
{final_f1:.6f}

Confusion Matrix:

TN = {tn}
FP = {fp}
FN = {fn}
TP = {tp}


============================================================
Q4 - SMOTE FROM SCRATCH
============================================================

Original minority samples:
{len(minority)}

Synthetic minority samples:
{len(synthetic_minority)}

SMOTE creates synthetic observations by interpolating
between minority observations and their nearest
minority neighbors.


============================================================
Q5 - BALANCED RF VS RF + SMOTE
============================================================

Best model:
{best_q5_model}

Best mean PR-AUC:
{best_q5_score:.6f}


============================================================
OUTPUT FOLDERS
============================================================

Q1:
output/Q1/

Q2:
output/Q2/

Q3:
output/Q3/

Q4:
output/Q4/

Q5:
output/Q5/

Final summary:
output/final_summary.csv
output/final_summary.txt

============================================================
END
============================================================
"""


with open(
    os.path.join(
        OUTPUT_DIR,
        "final_summary.txt"
    ),
    "w"
) as file:

    file.write(
        final_text
    )


# ================================================================
# FINAL CONSOLE MESSAGE
# ================================================================

print("\n" + "=" * 80)
print("ALL FIVE QUESTIONS COMPLETED")
print("=" * 80)

print("\nResults saved in:")

print(
    f"  {Q1_DIR}/"
)

print(
    f"  {Q2_DIR}/"
)

print(
    f"  {Q3_DIR}/"
)

print(
    f"  {Q4_DIR}/"
)

print(
    f"  {Q5_DIR}/"
)

print(
    f"\nFinal summary:"
)

print(
    f"  {OUTPUT_DIR}/final_summary.csv"
)

print(
    f"  {OUTPUT_DIR}/final_summary.txt"
)

print("\nDone.")
        
