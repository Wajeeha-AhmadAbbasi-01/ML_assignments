import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    accuracy_score
)


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("Unsupervised_ML/data/Telco_customer_churn copy.csv")


# =========================================================
# 2. CLEAN DATA
# =========================================================

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

df.loc[
    df["Total Charges"].isnull()
    & (df["Tenure Months"] == 0),
    "Total Charges"
] = 0


# =========================================================
# 3. FEATURES AND TARGET
# =========================================================

features = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Contract"
]

X = df[features].copy()

y = df["Churn Label"].map({
    "No": 0,
    "Yes": 1
})


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# =========================================================
# 5. FEATURE TYPES
# =========================================================

numeric_features = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges"
]

categorical_features = [
    "Contract"
]


# =========================================================
# 6. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# =========================================================
# 7. PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),

        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# =========================================================
# 8. HYPERPARAMETER GRID
# =========================================================

param_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10
    ]
}


# =========================================================
# 9. GRID SEARCH
# =========================================================

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)


# =========================================================
# 10. BEST PIPELINE
# =========================================================

best_pipeline = grid.best_estimator_

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest CV F1:")
print(grid.best_score_)


# =========================================================
# 11. TEST SET PREDICTIONS
# =========================================================

y_pred = best_pipeline.predict(X_test)

y_prob = best_pipeline.predict_proba(
    X_test
)[:, 1]


# =========================================================
# 12. TEST METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)


# =========================================================
# 13. DISPLAY METRICS
# =========================================================

print("\n==============================")
print("TEST SET RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


# =========================================================
# 14. SAVE PIPELINE
# =========================================================

joblib.dump(
    best_pipeline,
    "Unsupervised_ML/churn_pipeline.joblib"
)

print("\nPipeline saved:")
print("churn_pipeline.joblib")


# =========================================================
# 15. SAVE METRICS
# =========================================================

metrics = {
    "model": "Logistic Regression",
    "best_parameters": grid.best_params_,
    "cv_f1": float(grid.best_score_),
    "test_accuracy": float(accuracy),
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "test_pr_auc": float(pr_auc),
    "features": features
}

with open(
    "Unsupervised_ML/model_metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

print("\nMetrics saved:")
print("model_metrics.json")


# =========================================================
# 16. RELOAD PIPELINE
# =========================================================

loaded_pipeline = joblib.load(
    "Unsupervised_ML/churn_pipeline.joblib"
)

loaded_probabilities = loaded_pipeline.predict_proba(
    X_test
)[:, 1]


# =========================================================
# 17. VERIFY SAME PREDICTIONS
# =========================================================

same_predictions = (
    y_prob == loaded_probabilities
).all()

print("\n==============================")
print("PIPELINE VERIFICATION")
print("==============================")

print(
    "Reloaded pipeline gives identical predictions:",
    same_predictions
)