"""
Feature Scaling Experiments: Q1-Q5
Dataset: Breast Cancer Wisconsin dataset (built into scikit-learn)

Experiments:
Q1. KNN accuracy with vs. without feature scaling
Q2. PCA explained-variance ratios with vs. without scaling
Q3. ColumnTransformer with different scalers + cross-validation
Q4. Ridge coefficient stability across bootstrap samples
Q5. Logistic Regression (SAGA) convergence iterations with vs. without scaling

Install dependencies:
    pip install numpy pandas matplotlib scikit-learn

Run:
    python feature_scaling_experiments.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Save plots to files without opening plot windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer


OUTPUT_DIR = Path("feature_scaling_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_plot(filename):
    """Save the current plot to the output folder."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_table(df, filename):
    """Save a results table as CSV."""
    df.to_csv(OUTPUT_DIR / filename, index=False)


def load_dataset():
    """Load the dataset and create deliberately mixed feature scales."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()

    # Multiply every third feature by 1000 to create mixed numerical scales.
    X_mixed = X.copy()
    X_mixed.iloc[:, ::3] = X_mixed.iloc[:, ::3] * 1000

    return X_mixed, y


def question_1_knn(X_train, X_test, y_train, y_test):
    """Compare KNN test accuracy with and without scaling."""
    print("\n" + "=" * 65)
    print("Q1. KNN accuracy: without scaling vs. with scaling")
    print("=" * 65)

    # Without scaling
    knn_raw = KNeighborsClassifier(n_neighbors=5)
    knn_raw.fit(X_train, y_train)
    accuracy_raw = accuracy_score(y_test, knn_raw.predict(X_test))

    # With StandardScaler; the pipeline learns scaling from training data.
    knn_scaled = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])
    knn_scaled.fit(X_train, y_train)
    accuracy_scaled = accuracy_score(y_test, knn_scaled.predict(X_test))

    print(f"Accuracy without scaling: {accuracy_raw:.4f}")
    print(f"Accuracy with scaling:    {accuracy_scaled:.4f}")

    save_table(pd.DataFrame({
        "Method": ["Without scaling", "With StandardScaler"],
        "Test accuracy": [accuracy_raw, accuracy_scaled],
    }), "q1_knn_accuracy.csv")

    plt.figure(figsize=(7, 4))
    plt.bar(
        ["Without scaling", "With scaling"],
        [accuracy_raw, accuracy_scaled],
    )
    plt.ylim(0, 1)
    plt.ylabel("Test accuracy")
    plt.title("Q1: KNN Accuracy Comparison")
    save_plot("q1_knn_accuracy.png")


def question_2_pca(X_train):
    """Compare PCA explained-variance ratios with and without scaling."""
    print("\n" + "=" * 65)
    print("Q2. PCA explained variance: without scaling vs. with scaling")
    print("=" * 65)

    pca_raw = PCA()
    pca_raw.fit(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca_scaled = PCA()
    pca_scaled.fit(X_train_scaled)

    n_components = min(10, X_train.shape[1])
    components = np.arange(1, n_components + 1)

    print("First five explained-variance ratios without scaling:")
    print(np.round(pca_raw.explained_variance_ratio_[:5], 4))
    print("First five explained-variance ratios with scaling:")
    print(np.round(pca_scaled.explained_variance_ratio_[:5], 4))

    save_table(pd.DataFrame({
        "Component": np.arange(1, len(pca_raw.explained_variance_ratio_) + 1),
        "Variance ratio without scaling": pca_raw.explained_variance_ratio_,
        "Variance ratio with StandardScaler": pca_scaled.explained_variance_ratio_,
    }), "q2_pca_explained_variance.csv")
    print(
        "Cumulative variance in first five components "
        f"(without scaling): {pca_raw.explained_variance_ratio_[:5].sum():.4f}"
    )
    print(
        "Cumulative variance in first five components "
        f"(with scaling):    {pca_scaled.explained_variance_ratio_[:5].sum():.4f}"
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        components,
        pca_raw.explained_variance_ratio_[:n_components],
        marker="o",
        label="Without scaling",
    )
    plt.plot(
        components,
        pca_scaled.explained_variance_ratio_[:n_components],
        marker="o",
        label="With StandardScaler",
    )
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("Q2: PCA Explained Variance Comparison")
    plt.xticks(components)
    plt.legend()
    save_plot("q2_pca_explained_variance.png")


def question_3_column_transformer_cv(X, y):
    """Use different scalers for different column groups in cross-validation."""
    print("\n" + "=" * 65)
    print("Q3. ColumnTransformer + Pipeline + cross-validation")
    print("=" * 65)

    columns = X.columns.tolist()
    standard_columns = columns[:10]
    minmax_columns = columns[10:20]
    robust_columns = columns[20:]

    preprocessor = ColumnTransformer([
        ("standard", StandardScaler(), standard_columns),
        ("minmax", MinMaxScaler(), minmax_columns),
        ("robust", RobustScaler(), robust_columns),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])

    # cross_val_score fits each preprocessing step inside each training fold.
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print("Accuracy for each fold:", np.round(scores, 4))
    print(f"Mean cross-validation accuracy: {scores.mean():.4f}")
    print(f"Standard deviation:             {scores.std():.4f}")

    save_table(pd.DataFrame({
        "Fold": np.arange(1, len(scores) + 1),
        "Accuracy": scores,
    }), "q3_cross_validation_scores.csv")
    save_table(pd.DataFrame({
        "Metric": ["Mean accuracy", "Standard deviation"],
        "Value": [scores.mean(), scores.std()],
    }), "q3_cross_validation_summary.csv")


def question_4_ridge_bootstrap(X, y, bootstrap_count=100, alpha=10):
    """Compare Ridge coefficient variability across bootstrap samples."""
    print("\n" + "=" * 65)
    print("Q4. Ridge coefficient stability across bootstrap samples")
    print("=" * 65)
    print(
        "Note: the binary target is treated as numeric for this Ridge "
        "regularization demonstration."
    )

    rng = np.random.default_rng(42)
    raw_coefficients = []
    scaled_coefficients = []

    for _ in range(bootstrap_count):
        # Sample rows with replacement.
        indices = rng.integers(0, len(X), size=len(X))
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]

        # Ridge without scaling.
        ridge_raw = Ridge(alpha=alpha)
        ridge_raw.fit(X_boot, y_boot)
        raw_coefficients.append(ridge_raw.coef_)

        # Ridge with scaling. Keep alpha identical.
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)

        ridge_scaled = Ridge(alpha=alpha)
        ridge_scaled.fit(X_boot_scaled, y_boot)

        # Convert scaled coefficients back to original feature units.
        coefficients_original_units = ridge_scaled.coef_ / scaler.scale_
        scaled_coefficients.append(coefficients_original_units)

    raw_coefficients = np.asarray(raw_coefficients)
    scaled_coefficients = np.asarray(scaled_coefficients)

    raw_std = raw_coefficients.std(axis=0)
    scaled_std = scaled_coefficients.std(axis=0)

    raw_mean_abs = np.abs(raw_coefficients.mean(axis=0))
    scaled_mean_abs = np.abs(scaled_coefficients.mean(axis=0))
    epsilon = 1e-8

    raw_relative = raw_std / (raw_mean_abs + epsilon)
    scaled_relative = scaled_std / (scaled_mean_abs + epsilon)

    print(f"Bootstrap samples: {bootstrap_count}")
    print(f"Ridge alpha for both models: {alpha}")
    print(
        "Median relative coefficient variability without scaling: "
        f"{np.median(raw_relative):.4f}"
    )
    print(
        "Median relative coefficient variability with scaling:    "
        f"{np.median(scaled_relative):.4f}"
    )

    # Show the ten features with the largest raw-model coefficient SD.
    top_indices = np.argsort(raw_std)[-10:]
    comparison = pd.DataFrame({
        "Feature": X.columns[top_indices],
        "Raw coefficient SD": raw_std[top_indices],
        "Scaled coefficient SD (original units)": scaled_std[top_indices],
    }).sort_values("Raw coefficient SD", ascending=False)

    print("\nTen features with highest coefficient SD in the unscaled model:")
    print(comparison.round(6).to_string(index=False))
    save_table(comparison, "q4_ridge_top_feature_variability.csv")
    save_table(pd.DataFrame({
        "Method": ["Without scaling", "With scaling (coefficients in original units)"],
        "Median relative coefficient variability": [
            np.median(raw_relative), np.median(scaled_relative)
        ],
    }), "q4_ridge_stability_summary.csv")

    plt.figure(figsize=(7, 4))
    plt.bar(
        ["Without scaling", "With scaling"],
        [np.median(raw_relative), np.median(scaled_relative)],
    )
    plt.ylabel("Median relative coefficient variability")
    plt.title("Q4: Ridge Bootstrap Coefficient Stability")
    save_plot("q4_ridge_bootstrap_stability.png")


def question_5_logistic_convergence(X_train, y_train):
    """Compare SAGA iteration counts with and without standardization."""
    print("\n" + "=" * 65)
    print("Q5. Logistic Regression SAGA convergence")
    print("=" * 65)

    settings = {
        "solver": "saga",
        "max_iter": 5000,
        "tol": 1e-3,
        "random_state": 42,
    }

    # Capture convergence warnings so we can report them clearly.
    with warnings.catch_warnings(record=True) as raw_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model_raw = LogisticRegression(**settings)
        model_raw.fit(X_train, y_train)

    with warnings.catch_warnings(record=True) as scaled_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model_scaled = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(**settings)),
        ])
        model_scaled.fit(X_train, y_train)

    iterations_raw = int(model_raw.n_iter_[0])
    iterations_scaled = int(
        model_scaled.named_steps["logistic"].n_iter_[0]
    )
    raw_convergence_warning = any(
        issubclass(w.category, ConvergenceWarning) for w in raw_warnings
    )
    scaled_convergence_warning = any(
        issubclass(w.category, ConvergenceWarning) for w in scaled_warnings
    )

    print("Without scaling:")
    print(f"  Iterations: {iterations_raw}")
    print(f"  Convergence warning: {raw_convergence_warning}")

    print("With StandardScaler:")
    print(f"  Iterations: {iterations_scaled}")
    print(f"  Convergence warning: {scaled_convergence_warning}")

    save_table(pd.DataFrame({
        "Method": ["Without scaling", "With StandardScaler"],
        "Solver iterations": [iterations_raw, iterations_scaled],
        "Convergence warning": [raw_convergence_warning, scaled_convergence_warning],
    }), "q5_logistic_convergence.csv")

    plt.figure(figsize=(7, 4))
    plt.bar(
        ["Without scaling", "With scaling"],
        [iterations_raw, iterations_scaled],
    )
    plt.ylabel("Solver iterations")
    plt.title("Q5: Logistic Regression Convergence")
    save_plot("q5_logistic_convergence.png")


def main():
    X, y = load_dataset()

    print("Feature Scaling Experiments (Q1-Q5)")
    print("-" * 65)
    print(f"Dataset rows: {X.shape[0]}")
    print(f"Dataset features: {X.shape[1]}")
    print("\nExample feature standard deviations:")
    print(X.iloc[:, :6].std().round(2).to_string())

    # Shared stratified split for Q1, Q2, and Q5.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    question_1_knn(X_train, X_test, y_train, y_test)
    question_2_pca(X_train)
    question_3_column_transformer_cv(X, y)
    question_4_ridge_bootstrap(X, y)
    question_5_logistic_convergence(X_train, y_train)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print("Q1: Compared KNN test accuracy with and without StandardScaler.")
    print("Q2: Compared PCA explained-variance ratios with and without scaling.")
    print("Q3: Used ColumnTransformer with three scalers inside a Pipeline,")
    print("    evaluated using five-fold cross-validation.")
    print("Q4: Compared Ridge coefficient variability across bootstrap samples")
    print("    using the same alpha, converting coefficients to original units.")
    print("Q5: Compared SAGA Logistic Regression iteration counts and warnings.")
    print("\nInterpret your actual printed results; scaling helps many models,")
    print("but the size of the effect depends on the data and algorithm.")

    summary = """FEATURE SCALING EXPERIMENTS — SUMMARY

Q1 — KNN accuracy
Compared test accuracy with raw features and StandardScaler. KNN uses distances, so large-scale features can dominate when scaling is not applied.

Q2 — PCA explained variance
Compared component-wise explained-variance ratios before and after StandardScaler. PCA is sensitive to feature variance, so large-scale features can dominate unscaled PCA.

Q3 — ColumnTransformer and cross-validation
Applied StandardScaler, MinMaxScaler, and RobustScaler to different feature groups inside a Pipeline. Five-fold cross-validation fits preprocessing within each training fold, reducing data leakage risk.

Q4 — Ridge bootstrap coefficient stability
Repeated bootstrap sampling and compared Ridge coefficient variability using the same alpha. Scaled coefficients were converted back to original feature units. Scaling changes the effective regularization applied to differently scaled features; it does not guarantee every coefficient becomes more stable. The binary target is treated as numeric for this demonstration.

Q5 — Logistic Regression convergence
Compared SAGA solver iteration counts and convergence warnings with and without StandardScaler. Scaling often improves optimization, but the result depends on the data and settings.

Overall conclusion
Scaling is especially important for distance-based models, PCA, regularized models, and many gradient-based optimizers. Always fit preprocessing on training data only, preferably inside a Pipeline. The CSV files and PNG plots in this folder contain the actual results from this run.
"""
    (OUTPUT_DIR / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(f"\nAll question outputs and SUMMARY.txt saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
