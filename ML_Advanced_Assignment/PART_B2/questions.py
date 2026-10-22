"""
Regularization Project: Ridge, Lasso, and ElasticNet
Run: python questions.py
All tables, plots, and summaries are saved under outputs/.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
OUTPUT = Path(__file__).resolve().parent / "outputs"
OUTPUT.mkdir(exist_ok=True, parents=True)

def save_summary(folder, title, lines):
    folder.mkdir(exist_ok=True, parents=True)
    (folder / "summary.txt").write_text(
        title + "\n" + "=" * len(title) + "\n\n" +
        "\n".join(lines) + "\n", encoding="utf-8"
    )

def make_correlated_data(n_samples=500, random_state=42):
    """Create two strongly correlated predictors plus several other predictors."""
    rng = np.random.default_rng(random_state)
    base = rng.normal(size=n_samples)
    x1 = base + rng.normal(scale=0.08, size=n_samples)
    x2 = base + rng.normal(scale=0.08, size=n_samples)
    x3 = rng.normal(size=n_samples)
    x4 = rng.normal(size=n_samples)
    x5 = rng.normal(size=n_samples)
    X = np.column_stack([x1, x2, x3, x4, x5])
    # Both correlated features contribute to the target.
    y = 3.0 * x1 + 3.0 * x2 + 1.5 * x3 + rng.normal(scale=1.0, size=n_samples)
    names = [f"feature_{i}" for i in range(1, X.shape[1] + 1)]
    return X, y, names

def q1_ridge_from_scratch():
    folder = OUTPUT / "Q1_Ridge_From_Scratch"
    folder.mkdir(parents=True, exist_ok=True)
    X, y, names = make_correlated_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    # Standardize using training data only. Add an intercept column.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_design = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_test_design = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    alpha = 1.0
    penalty = np.eye(X_train_design.shape[1]) * alpha
    penalty[0, 0] = 0  # Do not penalize the intercept.

    # Closed-form Ridge: (X'X + lambda I)^(-1) X'y
    beta_scratch = np.linalg.solve(
        X_train_design.T @ X_train_design + penalty,
        X_train_design.T @ y_train
    )
    pred_scratch = X_test_design @ beta_scratch

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    pred_sklearn = model.predict(X_test_scaled)

    rows = [
        ("intercept", beta_scratch[0], model.intercept_),
    ]
    for i, name in enumerate(names):
        rows.append((name, beta_scratch[i + 1], model.coef_[i]))
    coef_df = pd.DataFrame(rows, columns=["term", "scratch_coefficient", "sklearn_coefficient"])
    coef_df["absolute_difference"] = (
        coef_df["scratch_coefficient"] - coef_df["sklearn_coefficient"]
    ).abs()
    coef_df.to_csv(folder / "coefficient_comparison.csv", index=False)

    metrics = pd.DataFrame([
        {"model": "Ridge from scratch", "MSE": mean_squared_error(y_test, pred_scratch),
         "R2": r2_score(y_test, pred_scratch)},
        {"model": "scikit-learn Ridge", "MSE": mean_squared_error(y_test, pred_sklearn),
         "R2": r2_score(y_test, pred_sklearn)},
    ])
    metrics.to_csv(folder / "model_metrics.csv", index=False)

    max_diff = float(coef_df["absolute_difference"].max())
    save_summary(folder, "Q1 — Ridge regression from scratch", [
        f"Dataset: {len(X)} rows and {X.shape[1]} features; features 1 and 2 are highly correlated.",
        f"Ridge alpha: {alpha}. The intercept is not penalized.",
        f"Maximum absolute coefficient difference from scikit-learn: {max_diff:.8f}.",
        f"Scratch test MSE: {metrics.iloc[0]['MSE']:.4f}; scikit-learn test MSE: {metrics.iloc[1]['MSE']:.4f}.",
        "Interpretation: close coefficients and predictions validate the closed-form implementation.",
        "The matrix solve is used instead of explicitly calculating a matrix inverse for better numerical stability."
    ])
    print("Q1 complete")

def q2_lasso_stability():
    folder = OUTPUT / "Q2_Lasso_Feature_Stability"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    n, p = 360, 30
    X = rng.normal(size=(n, p))
    true_beta = np.zeros(p)
    true_beta[[0, 3, 7, 12, 18]] = [4.0, -3.5, 2.8, 2.2, -2.5]
    y = X @ true_beta + rng.normal(scale=1.5, size=n)
    names = [f"feature_{i+1}" for i in range(p)]

    frequencies = np.zeros(p, dtype=int)
    selected_rows = []
    test_scores = []
    n_splits = 20

    for split in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=100 + split
        )
        model = Pipeline([
            ("scale", StandardScaler()),
            ("lasso", LassoCV(
                alphas=np.logspace(-3, 1, 50),
                cv=5, max_iter=20000, random_state=RANDOM_STATE
            ))
        ])
        model.fit(X_train, y_train)
        coefs = model.named_steps["lasso"].coef_
        selected = np.abs(coefs) > 1e-6
        frequencies += selected.astype(int)
        test_scores.append({
            "split": split + 1,
            "selected_feature_count": int(selected.sum()),
            "chosen_alpha": float(model.named_steps["lasso"].alpha_),
            "test_MSE": mean_squared_error(y_test, model.predict(X_test)),
            "test_R2": r2_score(y_test, model.predict(X_test)),
        })
        for i, name in enumerate(names):
            selected_rows.append({
                "split": split + 1, "feature": name,
                "selected": bool(selected[i]), "coefficient": float(coefs[i]),
                "is_true_signal": bool(true_beta[i] != 0)
            })

    stability = pd.DataFrame({
        "feature": names,
        "selection_count": frequencies,
        "selection_frequency": frequencies / n_splits,
        "is_true_signal": true_beta != 0,
        "true_coefficient": true_beta
    }).sort_values(["selection_frequency", "feature"], ascending=[False, True])
    stability.to_csv(folder / "feature_stability.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(folder / "selection_by_split.csv", index=False)
    pd.DataFrame(test_scores).to_csv(folder / "split_metrics.csv", index=False)

    plot_df = stability.sort_values("selection_frequency")
    plt.figure(figsize=(9, 8))
    plt.barh(plot_df["feature"], plot_df["selection_frequency"])
    plt.xlabel("Selection frequency across 20 splits")
    plt.ylabel("Feature")
    plt.title("LassoCV feature-selection stability")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(folder / "feature_stability.png", dpi=160)
    plt.close()

    stable = stability[stability["selection_frequency"] >= 0.8]["feature"].tolist()
    signal_names = [names[i] for i in np.where(true_beta != 0)[0]]
    save_summary(folder, "Q2 — LassoCV feature-selection stability", [
        f"Repeated random train/test splits: {n_splits}. Each split tunes alpha using 5-fold CV on training data only.",
        f"Features selected in at least 80% of splits: {', '.join(stable) if stable else 'None'}.",
        f"True signal features in this synthetic dataset: {', '.join(signal_names)}.",
        "Interpretation: high selection frequency suggests a feature is selected robustly across changes in the training sample.",
        "A feature can be useful but unstable when predictors overlap or the signal is weak; selection frequency is not proof of causality.",
        f"Mean held-out R2 across splits: {np.mean([r['test_R2'] for r in test_scores]):.4f}."
    ])
    print("Q2 complete")

def q3_regularization_paths():
    folder = OUTPUT / "Q3_Regularization_Paths"
    folder.mkdir(parents=True, exist_ok=True)
    X, y, names = make_correlated_data(n_samples=350, random_state=21)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Center y so intercept handling is simple and coefficient paths are comparable.
    y = y - y.mean()
    alphas = np.logspace(-3, 2, 70)
    models = {
        "Ridge": lambda a: Ridge(alpha=a),
        "Lasso": lambda a: Lasso(alpha=a, max_iter=20000),
        "ElasticNet (l1_ratio=0.5)": lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=20000),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    path_rows = []
    for ax, (model_name, factory) in zip(axes, models.items()):
        coefficients = []
        for alpha in alphas:
            model = factory(alpha)
            model.fit(X, y)
            coefficients.append(model.coef_.copy())
            for i, name in enumerate(names):
                path_rows.append({
                    "model": model_name, "alpha": alpha,
                    "log10_alpha": np.log10(alpha),
                    "feature": name, "coefficient": model.coef_[i]
                })
        coefficients = np.asarray(coefficients)
        for i, name in enumerate(names):
            ax.plot(np.log10(alphas), coefficients[:, i], label=name)
        ax.set_title(model_name)
        ax.set_xlabel("log10(alpha)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Coefficient value")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Regularization paths on the same dataset")
    fig.tight_layout()
    fig.savefig(folder / "regularization_paths.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(path_rows).to_csv(folder / "regularization_path_values.csv", index=False)

    save_summary(folder, "Q3 — Regularization paths", [
        "All three models use the same standardized predictors and target, and the same alpha grid.",
        "Ridge: coefficients usually shrink smoothly toward zero, but typically do not become exactly zero.",
        "Lasso: some coefficients become exactly zero, creating a sparse model and performing feature selection.",
        "ElasticNet: combines L1 and L2 penalties; it can set coefficients to zero while often keeping correlated predictors more jointly.",
        "As alpha increases, regularization becomes stronger. The precise path depends on the data and the scaling."
    ])
    print("Q3 complete")

def q4_nested_cv():
    folder = OUTPUT / "Q4_Nested_Cross_Validation"
    folder.mkdir(parents=True, exist_ok=True)
    X, y, names = make_correlated_data(n_samples=420, random_state=33)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + 1)
    alpha_values = np.logspace(-3, 1, 18)
    l1_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(max_iter=20000))
        ])
        search = GridSearchCV(
            pipeline,
            param_grid={
                "model__alpha": alpha_values,
                "model__l1_ratio": l1_values
            },
            scoring="neg_mean_squared_error",
            cv=inner_cv,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        prediction = search.predict(X_test)
        fold_rows.append({
            "outer_fold": fold,
            "test_MSE": mean_squared_error(y_test, prediction),
            "test_R2": r2_score(y_test, prediction),
            "best_alpha": search.best_params_["model__alpha"],
            "best_l1_ratio": search.best_params_["model__l1_ratio"],
            "inner_best_MSE": -search.best_score_
        })

    results = pd.DataFrame(fold_rows)
    results.to_csv(folder / "outer_fold_results.csv", index=False)
    results[["outer_fold", "test_MSE", "test_R2"]].to_csv(
        folder / "outer_test_metrics.csv", index=False
    )
    mean_mse = results["test_MSE"].mean()
    std_mse = results["test_MSE"].std(ddof=1)
    mean_r2 = results["test_R2"].mean()
    save_summary(folder, "Q4 — Nested cross-validation for ElasticNet", [
        "Outer loop: 5 folds estimates generalization performance on held-out data.",
        "Inner loop: 4-fold GridSearchCV tunes alpha and l1_ratio using only each outer training fold.",
        "StandardScaler is inside the pipeline, so it is fitted only on the relevant training fold.",
        f"Mean outer-fold MSE: {mean_mse:.4f} (standard deviation {std_mse:.4f}).",
        f"Mean outer-fold R2: {mean_r2:.4f}.",
        f"Best alpha values by outer fold: {', '.join(f'{x:.4g}' for x in results['best_alpha'])}.",
        f"Best l1_ratio values by outer fold: {', '.join(f'{x:.2f}' for x in results['best_l1_ratio'])}.",
        "Interpretation: report the mean outer score as the nested-CV performance estimate; do not report the inner best score as the unbiased test estimate."
    ])
    print("Q4 complete")

def q5_grouping_effect():
    folder = OUTPUT / "Q5_ElasticNet_Grouping_Effect"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(77)
    n = 500
    shared_a = rng.normal(size=n)
    shared_b = rng.normal(size=n)

    # Two correlated groups: group A (features 1-4) and group B (features 5-8).
    X = np.column_stack([
        shared_a + rng.normal(scale=0.12, size=n) for _ in range(4)
    ] + [
        shared_b + rng.normal(scale=0.12, size=n) for _ in range(4)
    ] + [
        rng.normal(size=n) for _ in range(4)
    ])
    names = [f"feature_{i+1}" for i in range(X.shape[1])]
    y = 2.0 * X[:, :4].mean(axis=1) + 2.0 * X[:, 4:8].mean(axis=1) + rng.normal(scale=0.8, size=n)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use fixed penalties to make a direct, reproducible comparison.
    lasso = Lasso(alpha=0.5, max_iter=20000).fit(X_train_scaled, y_train)
    elastic = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=20000).fit(X_train_scaled, y_train)

    comparison = pd.DataFrame({
        "feature": names,
        "group": ["Group A"] * 4 + ["Group B"] * 4 + ["Noise"] * 4,
        "lasso_coefficient": lasso.coef_,
        "elasticnet_coefficient": elastic.coef_,
    })
    comparison["lasso_selected"] = comparison["lasso_coefficient"].abs() > 1e-6
    comparison["elasticnet_selected"] = comparison["elasticnet_coefficient"].abs() > 1e-6
    comparison.to_csv(folder / "feature_selection_comparison.csv", index=False)

    group_summary = comparison.groupby("group", sort=False).agg(
        lasso_selected_count=("lasso_selected", "sum"),
        elasticnet_selected_count=("elasticnet_selected", "sum"),
        lasso_mean_abs_coefficient=("lasso_coefficient", lambda x: np.abs(x).mean()),
        elasticnet_mean_abs_coefficient=("elasticnet_coefficient", lambda x: np.abs(x).mean()),
    ).reset_index()
    group_summary.to_csv(folder / "group_summary.csv", index=False)

    x_pos = np.arange(len(names))
    width = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar(x_pos - width/2, comparison["lasso_coefficient"], width, label="Lasso")
    plt.bar(x_pos + width/2, comparison["elasticnet_coefficient"], width, label="ElasticNet")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(x_pos, names, rotation=45)
    plt.ylabel("Coefficient")
    plt.title("Lasso vs ElasticNet on correlated feature groups")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "grouping_effect.png", dpi=160)
    plt.close()

    lasso_a = int(comparison.iloc[:4]["lasso_selected"].sum())
    elastic_a = int(comparison.iloc[:4]["elasticnet_selected"].sum())
    lasso_b = int(comparison.iloc[4:8]["lasso_selected"].sum())
    elastic_b = int(comparison.iloc[4:8]["elasticnet_selected"].sum())
    metrics = pd.DataFrame([
        {"model": "Lasso", "test_MSE": mean_squared_error(y_test, lasso.predict(X_test_scaled)),
         "test_R2": r2_score(y_test, lasso.predict(X_test_scaled))},
        {"model": "ElasticNet", "test_MSE": mean_squared_error(y_test, elastic.predict(X_test_scaled)),
         "test_R2": r2_score(y_test, elastic.predict(X_test_scaled))},
    ])
    metrics.to_csv(folder / "model_metrics.csv", index=False)

    save_summary(folder, "Q5 — ElasticNet grouping effect", [
        "Synthetic data contains two groups of four highly correlated signal features plus four noise features.",
        f"Group A selected: Lasso {lasso_a}/4; ElasticNet {elastic_a}/4.",
        f"Group B selected: Lasso {lasso_b}/4; ElasticNet {elastic_b}/4.",
        f"Test MSE — Lasso: {metrics.iloc[0]['test_MSE']:.4f}; ElasticNet: {metrics.iloc[1]['test_MSE']:.4f}.",
        "Expected pattern: Lasso often selects one or a few substitutes from a correlated group; ElasticNet's L2 component can spread weight across correlated predictors.",
        "This is an empirical demonstration, not a guarantee: results depend on the data, alpha, and l1_ratio."
    ])
    print("Q5 complete")

def main():
    q1_ridge_from_scratch()
    q2_lasso_stability()
    q3_regularization_paths()
    q4_nested_cv()
    q5_grouping_effect()

    summaries = []
    for qfolder in sorted(OUTPUT.iterdir()):
        summary = qfolder / "summary.txt"
        if summary.exists():
            summaries.append(summary.read_text(encoding="utf-8"))
    (OUTPUT / "overall_summary.txt").write_text(
        "REGULARIZATION PROJECT — OVERALL SUMMARY\n"
        "========================================\n\n" + "\n\n".join(summaries),
        encoding="utf-8"
    )
    print(f"\nAll questions complete. Outputs saved to: {OUTPUT}")

if __name__ == "__main__":
    main()
