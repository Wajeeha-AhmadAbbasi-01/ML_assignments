# ================================================================
# PROJECT — Diagnosing and Repairing Missingness in EHR Data
# A1 — Handling Missing Data
#
# This single Python file:
# 1. Loads the Pima Indians Diabetes dataset
# 2. Converts clinically impossible zeros to missing values
# 3. Simulates MCAR, MAR, and MNAR missingness
# 4. Profiles missingness
# 5. Analyzes missingness vs target
# 6. Builds a leakage-free feature-specific pipeline
# 7. Evaluates the final model
# 8. Runs a 4-strategy ablation study
# 9. Saves ALL tables, plots, model, and summary files
#    inside an "outputs" folder
# ================================================================

# -----------------------------
# 1. IMPORTS
# -----------------------------
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# -----------------------------
# 2. SETTINGS
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

np.random.seed(RANDOM_STATE)

# Create output folder automatically
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EHR MISSING DATA PROJECT")
print("=" * 70)
print(f"Output folder: {OUTPUT_DIR.resolve()}")


# -----------------------------
# 3. LOAD DATASET
# -----------------------------
print("\n" + "=" * 70)
print("1. LOADING DATA")
print("=" * 70)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

df = pd.read_csv(url, names=columns)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nTarget distribution:")
print(df["Outcome"].value_counts())
print("\nTarget proportion:")
print(df["Outcome"].value_counts(normalize=True))


# Save original data
df.to_csv(OUTPUT_DIR / "01_original_pima_data.csv", index=False)


# -----------------------------
# 4. CONVERT IMPOSSIBLE ZEROS
# -----------------------------
print("\n" + "=" * 70)
print("2. HANDLING CLINICALLY IMPOSSIBLE ZERO VALUES")
print("=" * 70)

# Zero is meaningful for Pregnancies, so it is NOT converted.
zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

df_ehr = df.copy()

for col in zero_as_missing:
    df_ehr[col] = df_ehr[col].replace(0, np.nan)

print("Missing values after converting impossible zeros:")
print(df_ehr.isnull().sum())

df_ehr.to_csv(
    OUTPUT_DIR / "02_ehr_with_baseline_missing_values.csv",
    index=False
)


# -----------------------------
# 5. SEPARATE FEATURES AND TARGET
# -----------------------------
target = "Outcome"

X_base = df_ehr.drop(columns=[target])
y = df_ehr[target]


# ================================================================
# 6. CREATE COMPLETE REFERENCE FOR CONTROLLED SIMULATION
# ================================================================
print("\n" + "=" * 70)
print("3. CREATING REFERENCE DATA FOR MCAR / MAR / MNAR SIMULATION")
print("=" * 70)

# The original Pima data contains impossible zero values.
# For controlled simulation, we create a temporary complete reference
# by median-filling those baseline missing values.
#
# IMPORTANT:
# This reference is ONLY used to generate known missingness mechanisms.
# It is NOT used as a pre-fitted imputer for the final model.

X_complete = X_base.copy()

for col in X_complete.columns:
    X_complete[col] = X_complete[col].fillna(X_complete[col].median())


# -----------------------------
# 7. INJECT MCAR, MAR, MNAR
# -----------------------------
print("\n" + "=" * 70)
print("4. SIMULATING MCAR, MAR, AND MNAR")
print("=" * 70)

rng = np.random.RandomState(RANDOM_STATE)

X_project = X_complete.copy()

# ------------------------------------------------
# MCAR
# SkinThickness missing completely at random
# ------------------------------------------------
mcar_rate = 0.15

mcar_mask = rng.rand(len(X_project)) < mcar_rate

X_project.loc[mcar_mask, "SkinThickness"] = np.nan


# ------------------------------------------------
# MAR
# BMI missingness depends on observed Age
# ------------------------------------------------
age = X_complete["Age"]

age_standardized = (
    (age - age.mean()) / age.std()
)

prob_mar = (
    0.05
    + 0.20 / (1 + np.exp(-age_standardized))
)

prob_mar = np.clip(prob_mar, 0, 0.30)

mar_mask = rng.rand(len(X_project)) < prob_mar

X_project.loc[mar_mask, "BMI"] = np.nan


# ------------------------------------------------
# MNAR
# Insulin missingness depends on true Insulin
# ------------------------------------------------
insulin = X_complete["Insulin"]

insulin_standardized = (
    (insulin - insulin.mean()) / insulin.std()
)

prob_mnar = (
    0.05
    + 0.25 / (1 + np.exp(-insulin_standardized))
)

prob_mnar = np.clip(prob_mnar, 0, 0.35)

mnar_mask = rng.rand(len(X_project)) < prob_mnar

X_project.loc[mnar_mask, "Insulin"] = np.nan


# Final project dataset
project_df = X_project.copy()
project_df[target] = y.values

print("Final project shape:", project_df.shape)

print("\nMissing values:")
print(project_df.isnull().sum())

project_df.to_csv(
    OUTPUT_DIR / "03_simulated_ehr_dataset.csv",
    index=False
)


# -----------------------------
# 8. MISSINGNESS PROFILE
# -----------------------------
print("\n" + "=" * 70)
print("5. MISSINGNESS PROFILE")
print("=" * 70)

missing_profile = pd.DataFrame({
    "Missing Count": project_df.isnull().sum(),
    "Missing Rate (%)": project_df.isnull().mean() * 100,
})

missing_profile = missing_profile.sort_values(
    "Missing Rate (%)",
    ascending=False
)

print(missing_profile)

missing_profile.to_csv(
    OUTPUT_DIR / "04_missingness_profile.csv"
)


# Plot missingness rate
plt.figure(figsize=(10, 6))

missing_profile["Missing Rate (%)"].plot(
    kind="bar"
)

plt.title("Missingness Rate by Feature")
plt.ylabel("Missing Values (%)")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_missingness_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------
# 9. MISSINGNESS PATTERN HEATMAP
# -----------------------------
print("\nCreating missingness pattern heatmap...")

plt.figure(figsize=(12, 7))

sns.heatmap(
    project_df.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missingness Pattern")
plt.xlabel("Features")
plt.ylabel("Observations")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_missingness_pattern_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------
# 10. MISSINGNESS CORRELATION
# -----------------------------
print("\nCalculating missingness correlations...")

missing_indicators = (
    project_df.drop(columns=[target])
    .isnull()
    .astype(int)
)

missing_corr = missing_indicators.corr()

print("\nMissingness correlation matrix:")
print(missing_corr.round(3))

missing_corr.to_csv(
    OUTPUT_DIR / "07_missingness_correlation.csv"
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    missing_corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)

plt.title("Correlation Between Missingness Indicators")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_missingness_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 11. MISSINGNESS VS TARGET
# ================================================================
print("\n" + "=" * 70)
print("6. MISSINGNESS VS TARGET")
print("=" * 70)

target_missingness_analysis = []

for col in X_project.columns:

    missing = X_project[col].isnull()

    if missing.sum() > 0:

        missing_target_rate = y[missing].mean()
        observed_target_rate = y[~missing].mean()

        target_missingness_analysis.append({
            "Feature": col,
            "Missing Count": int(missing.sum()),
            "Missing Target Rate": missing_target_rate,
            "Observed Target Rate": observed_target_rate,
            "Difference": (
                missing_target_rate
                - observed_target_rate
            ),
        })

target_missingness_df = pd.DataFrame(
    target_missingness_analysis
)

print(target_missingness_df.round(4))

target_missingness_df.to_csv(
    OUTPUT_DIR / "09_missingness_vs_target.csv",
    index=False
)


# Plot target rate by missingness
if not target_missingness_df.empty:

    plt.figure(figsize=(10, 6))

    x = np.arange(len(target_missingness_df))
    width = 0.35

    plt.bar(
        x - width / 2,
        target_missingness_df["Missing Target Rate"],
        width,
        label="Missing"
    )

    plt.bar(
        x + width / 2,
        target_missingness_df["Observed Target Rate"],
        width,
        label="Observed"
    )

    plt.xticks(
        x,
        target_missingness_df["Feature"],
        rotation=45
    )

    plt.ylabel("Diabetes Outcome Rate")
    plt.xlabel("Feature")
    plt.title("Outcome Rate: Missing vs Observed")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "10_missingness_vs_target.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# 12. DOCUMENT THE KNOWN SIMULATION MECHANISMS
# ================================================================
print("\n" + "=" * 70)
print("7. MISSINGNESS MECHANISM DIAGNOSIS")
print("=" * 70)

diagnosis_table = pd.DataFrame({
    "Feature": [
        "SkinThickness",
        "BMI",
        "Insulin"
    ],
    "Injected Mechanism": [
        "MCAR",
        "MAR",
        "MNAR"
    ],
    "How Missingness Was Generated": [
        "Randomly, independent of observed and unobserved values",
        "Missingness probability depends on observed Age",
        "Missingness probability depends on the underlying Insulin value"
    ],
    "Chosen Repair": [
        "Median imputation",
        "MICE / Iterative Imputation",
        "Median imputation + missing indicator"
    ],
})

print(diagnosis_table.to_string(index=False))

diagnosis_table.to_csv(
    OUTPUT_DIR / "11_missingness_diagnosis.csv",
    index=False
)


# ================================================================
# 13. FEATURE-SPECIFIC LEAKAGE-FREE PIPELINE
# ================================================================
print("\n" + "=" * 70)
print("8. BUILDING FINAL LEAKAGE-FREE PIPELINE")
print("=" * 70)

# MCAR feature
mice_features = [
    "BMI"
]

# MNAR feature
median_indicator_features = [
    "Insulin"
]

# Other features / MCAR feature
median_features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "DiabetesPedigreeFunction",
    "Age",
]


# MICE pipeline
mice_transformer = Pipeline([
    (
        "imputer",
        IterativeImputer(
            max_iter=10,
            random_state=RANDOM_STATE
        )
    )
])


# Median + missingness indicator
median_indicator_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    )
])


# Regular median imputation
median_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


# Combine feature-specific preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "mice",
            mice_transformer,
            mice_features
        ),
        (
            "median_indicator",
            median_indicator_transformer,
            median_indicator_features
        ),
        (
            "median",
            median_transformer,
            median_features
        ),
    ],
    remainder="drop"
)


# Complete final pipeline
final_pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        )
    ),
])


print("Final pipeline created successfully.")


# ================================================================
# 14. TRAIN / TEST SPLIT
# ================================================================
print("\n" + "=" * 70)
print("9. TRAIN / TEST EVALUATION")
print("=" * 70)

X = project_df.drop(columns=[target])
y = project_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# Fit pipeline
final_pipeline.fit(
    X_train,
    y_train
)


# Predict
test_predictions = final_pipeline.predict(X_test)

test_probabilities = final_pipeline.predict_proba(
    X_test
)[:, 1]


# Metrics
test_auc = roc_auc_score(
    y_test,
    test_probabilities
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)


test_results = pd.DataFrame({
    "Metric": [
        "ROC-AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ],
    "Value": [
        test_auc,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1
    ]
})

print("\nTest-set results:")
print(test_results.round(4).to_string(index=False))

test_results.to_csv(
    OUTPUT_DIR / "12_final_test_metrics.csv",
    index=False
)


# Classification report
report = classification_report(
    y_test,
    test_predictions,
    output_dict=True,
    zero_division=0
)

classification_report_df = pd.DataFrame(report).transpose()

classification_report_df.to_csv(
    OUTPUT_DIR / "13_classification_report.csv"
)


# Confusion matrix
cm = confusion_matrix(
    y_test,
    test_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

cm_df.to_csv(
    OUTPUT_DIR / "14_confusion_matrix.csv"
)


plt.figure(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes", "Diabetes"]
)

disp.plot(
    values_format="d"
)

plt.title("Final Model Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "15_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 15. CROSS-VALIDATION OF FINAL PIPELINE
# ================================================================
print("\n" + "=" * 70)
print("10. FINAL PIPELINE 5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

final_cv_scores = cross_val_score(
    final_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

final_cv_results = pd.DataFrame({
    "Fold": np.arange(1, N_SPLITS + 1),
    "ROC-AUC": final_cv_scores
})

print(final_cv_results.round(4).to_string(index=False))

print(
    "\nMean CV ROC-AUC:",
    round(final_cv_scores.mean(), 4)
)

print(
    "Std CV ROC-AUC:",
    round(final_cv_scores.std(), 4)
)

final_cv_results.to_csv(
    OUTPUT_DIR / "16_final_pipeline_cv_scores.csv",
    index=False
)

pd.DataFrame({
    "Metric": [
        "Mean CV ROC-AUC",
        "Std CV ROC-AUC"
    ],
    "Value": [
        final_cv_scores.mean(),
        final_cv_scores.std()
    ]
}).to_csv(
    OUTPUT_DIR / "17_final_pipeline_cv_summary.csv",
    index=False
)


# ================================================================
# 16. ABLATION STUDY
# ================================================================
print("\n" + "=" * 70)
print("11. ABLATION STUDY")
print("=" * 70)

# ------------------------------------------------
# Strategy 1 — Mean imputation
# ------------------------------------------------
mean_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="mean")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        )
    )
])


# ------------------------------------------------
# Strategy 2 — Median imputation
# ------------------------------------------------
median_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        )
    )
])


# ------------------------------------------------
# Strategy 3 — Median + missingness indicator
# ------------------------------------------------
indicator_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        )
    )
])


# ------------------------------------------------
# Strategy 4 — MICE
# ------------------------------------------------
mice_pipeline = Pipeline([
    (
        "imputer",
        IterativeImputer(
            max_iter=10,
            random_state=RANDOM_STATE
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        )
    )
])


ablation_models = {
    "Mean": mean_pipeline,
    "Median": median_pipeline,
    "Median + Indicator": indicator_pipeline,
    "MICE": mice_pipeline,
}


ablation_results = []
detailed_ablation_results = []


for name, model in ablation_models.items():

    print(f"\nRunning: {name}")

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )

    ablation_results.append({
        "Strategy": name,
        "Mean AUC": scores.mean(),
        "Std AUC": scores.std()
    })

    detailed_ablation_results.append({
        "Strategy": name,
        "Mean AUC": scores.mean(),
        "Std AUC": scores.std(),
        "Min AUC": scores.min(),
        "Max AUC": scores.max(),
        "Fold 1": scores[0],
        "Fold 2": scores[1],
        "Fold 3": scores[2],
        "Fold 4": scores[3],
        "Fold 5": scores[4],
    })


ablation_df = pd.DataFrame(
    ablation_results
)

detailed_ablation_df = pd.DataFrame(
    detailed_ablation_results
)


print("\nAblation results:")
print(
    ablation_df.round(4).to_string(index=False)
)


ablation_df.to_csv(
    OUTPUT_DIR / "18_ablation_results.csv",
    index=False
)

detailed_ablation_df.to_csv(
    OUTPUT_DIR / "19_detailed_ablation_results.csv",
    index=False
)


# ================================================================
# 17. ABLATION PLOT
# ================================================================
plt.figure(figsize=(10, 6))

sns.barplot(
    data=ablation_df,
    x="Strategy",
    y="Mean AUC"
)

plt.ylim(0.5, 1.0)

plt.title(
    "Ablation Study: Imputation Strategy vs ROC-AUC"
)

plt.ylabel("Mean 5-Fold ROC-AUC")
plt.xlabel("Imputation Strategy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "20_ablation_auc_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 18. SAVE FINAL PIPELINE
# ================================================================
print("\n" + "=" * 70)
print("12. SAVING FINAL PIPELINE")
print("=" * 70)

pipeline_path = (
    OUTPUT_DIR /
    "21_ehr_missingness_final_pipeline.joblib"
)

joblib.dump(
    final_pipeline,
    pipeline_path
)

print("Saved:", pipeline_path)


# ================================================================
# 19. SAVE SIMULATION INFORMATION
# ================================================================
simulation_info = pd.DataFrame({
    "Feature": [
        "SkinThickness",
        "BMI",
        "Insulin"
    ],
    "Mechanism": [
        "MCAR",
        "MAR",
        "MNAR"
    ],
    "Mechanism_definition": [
        "Missingness probability is independent of data values",
        "Missingness probability depends on observed Age",
        "Missingness probability depends on the underlying Insulin value"
    ],
    "Rate_control": [
        f"Approximately {mcar_rate * 100:.0f}%",
        "Probability varies with Age, capped at 30%",
        "Probability varies with Insulin, capped at 35%"
    ]
})

simulation_info.to_csv(
    OUTPUT_DIR / "22_simulation_information.csv",
    index=False
)


# ================================================================
# 20. CREATE PROJECT SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("13. CREATING PROJECT SUMMARY")
print("=" * 70)

best_ablation_row = ablation_df.loc[
    ablation_df["Mean AUC"].idxmax()
]

summary_text = f"""
EHR MISSING DATA PROJECT SUMMARY
================================

Dataset:
Pima Indians Diabetes dataset

Rows:
{len(project_df)}

Features:
{len(X.columns)}

Target:
Outcome

Train/Test Split:
{int((1 - TEST_SIZE) * 100)}% / {int(TEST_SIZE * 100)}%

Cross-validation:
{N_SPLITS}-fold StratifiedKFold

Final Test ROC-AUC:
{test_auc:.4f}

Final Test Accuracy:
{test_accuracy:.4f}

Final Test Precision:
{test_precision:.4f}

Final Test Recall:
{test_recall:.4f}

Final Test F1:
{test_f1:.4f}

Final Pipeline Mean CV ROC-AUC:
{final_cv_scores.mean():.4f}

Final Pipeline CV ROC-AUC Std:
{final_cv_scores.std():.4f}

Best Ablation Strategy by Mean ROC-AUC:
{best_ablation_row["Strategy"]}

Best Ablation Mean ROC-AUC:
{best_ablation_row["Mean AUC"]:.4f}

Missingness strategies:
- SkinThickness: Median imputation for simulated MCAR
- BMI: MICE / IterativeImputer for simulated MAR
- Insulin: Median + missingness indicator for simulated MNAR
- Other variables: Median imputation

Important limitation:
The missingness mechanisms are known because MCAR, MAR, and MNAR
were deliberately injected for this project. In a real EHR dataset,
MAR and especially MNAR generally cannot be proven from observed data alone.

Leakage prevention:
All imputation and preprocessing steps are contained inside sklearn
Pipeline / ColumnTransformer objects and are therefore fitted separately
inside each cross-validation training fold.

Generated outputs are stored in:
{OUTPUT_DIR.resolve()}
"""

print(summary_text)

with open(
    OUTPUT_DIR / "23_project_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(summary_text)


# ================================================================
# 21. SAVE REQUIREMENTS / STRATEGY TABLE
# ================================================================
strategy_table = pd.DataFrame({
    "Feature": [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ],
    "Diagnosis": [
        "Baseline missingness",
        "Baseline missingness",
        "Baseline missingness",
        "Simulated MCAR",
        "Simulated MNAR",
        "Simulated MAR",
        "Baseline missingness",
        "Baseline missingness"
    ],
    "Strategy": [
        "Median",
        "Median",
        "Median",
        "Median",
        "Median + Indicator",
        "MICE",
        "Median",
        "Median"
    ],
    "Justification": [
        "Simple robust imputation",
        "Robust to skew and outliers",
        "Robust to skew and outliers",
        "MCAR does not require a special missingness indicator",
        "Indicator preserves information that the value was missing",
        "Uses relationships among observed variables under MAR assumption",
        "Simple robust imputation",
        "Simple robust imputation"
    ]
})

strategy_table.to_csv(
    OUTPUT_DIR / "24_feature_strategy_table.csv",
    index=False
)


# ================================================================
# 22. LIST OUTPUTS
# ================================================================
print("\n" + "=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)

print("\nAll generated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    if file.is_file():
        print(" -", file.name)

print("\nFinal test ROC-AUC:", round(test_auc, 4))
print("Mean CV ROC-AUC:", round(final_cv_scores.mean(), 4))
print("Best ablation:", best_ablation_row["Strategy"])
print("Best ablation AUC:", round(best_ablation_row["Mean AUC"], 4))

print("\nOutputs saved to:")
print(OUTPUT_DIR.resolve())

print("\nDone.")
