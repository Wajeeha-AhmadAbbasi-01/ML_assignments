"""Part A4: The Cost of Skipping Scaling.
Run: pip install numpy pandas matplotlib scikit-learn
     python cost_of_skipping_scaling.py
Outputs are saved in ./outputs.
"""
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

def metrics(model, Xtr, Xte, ytr, yte):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred, zero_division=0),
        "Recall": recall_score(yte, pred, zero_division=0),
        "F1": f1_score(yte, pred, zero_division=0),
        "Convergence warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
    }

def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data.copy(), data.target.copy()

    # Multiply every third feature by 1000 to create deliberately mixed scales.
    X.iloc[:, ::3] = X.iloc[:, ::3] * 1000
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Each pair compares the same model settings on raw and standardized data.
    models = {
        "KNN": (
            KNeighborsClassifier(n_neighbors=5),
            Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
        ),
        "RBF SVM": (
            SVC(kernel="rbf", C=1.0, gamma="scale"),
            Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", C=1.0, gamma="scale"))]),
        ),
        "Regularized Logistic Regression": (
            LogisticRegression(C=1.0, max_iter=5000, random_state=42),
            Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=42))]),
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]),
        ),
    }

    explanations = {
        "KNN": "KNN uses distances. Large-valued features can dominate Euclidean distance, changing which rows count as nearest neighbors.",
        "RBF SVM": "The RBF kernel depends on squared distances: exp(-gamma * ||x-z||^2). Large-scale features can dominate similarities and change the decision boundary.",
        "Regularized Logistic Regression": "L2 regularization penalizes coefficient squares. Feature units affect coefficient magnitudes, so the same penalty treats differently scaled features unevenly. Scaling can also improve optimization.",
        "Random Forest": "Trees split one feature at a time using thresholds. Positive rescaling preserves feature ordering, so equivalent splits remain possible. Tree ensembles usually do not need scaling.",
    }

    rows, effect_rows = [], []
    for name, (raw, scaled) in models.items():
        a = metrics(raw, Xtr, Xte, ytr, yte)
        b = metrics(scaled, Xtr, Xte, ytr, yte)
        rows.extend([{"Model": name, "Version": "Without scaling", **a},
                     {"Model": name, "Version": "With StandardScaler", **b}])
        da, df = b["Accuracy"] - a["Accuracy"], b["F1"] - a["F1"]
        if abs(da) >= .02 or abs(df) >= .02:
            label = "Substantial change (>= 0.02 in accuracy or F1)"
        elif abs(da) <= .005 and abs(df) <= .005:
            label = "Little/no measurable change (<= 0.005 in both)"
        else:
            label = "Small/moderate change"
        effect_rows.append({
            "Model": name, "Accuracy change (scaled - raw)": da,
            "F1 change (scaled - raw)": df, "Observed comparison": label,
            "Mechanistic explanation": explanations[name],
        })

    results = pd.DataFrame(rows)
    effects = pd.DataFrame(effect_rows)
    results.to_csv(OUT / "model_metrics_long.csv", index=False)
    wide = results.pivot(index="Model", columns="Version", values=["Accuracy", "Precision", "Recall", "F1"])
    wide.columns = [f"{metric} — {version}" for metric, version in wide.columns]
    wide.reset_index().to_csv(OUT / "results_comparison_table.csv", index=False)
    effects.to_csv(OUT / "scaling_effect_and_explanations.csv", index=False)

    print("\nRESULTS TABLE")
    print(wide.round(4).to_string())
    print("\nSCALING EFFECT")
    print(effects[["Model", "Accuracy change (scaled - raw)", "F1 change (scaled - raw)", "Observed comparison"]].round(4).to_string())

    for metric, filename, title in [
        ("Accuracy", "accuracy_comparison.png", "Test Accuracy: With vs Without Scaling"),
        ("F1", "f1_comparison.png", "F1 Score: With vs Without Scaling"),
    ]:
        chart = results.pivot(index="Model", columns="Version", values=metric)
        chart[["Without scaling", "With StandardScaler"]].plot(kind="bar", figsize=(10, 5))
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.title(title)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(OUT / filename, dpi=160)
        plt.close()

    summary = [
        "PART A4 — THE COST OF SKIPPING SCALING", "=" * 42, "",
        "Dataset: scikit-learn Breast Cancer Wisconsin dataset.",
        "Every third feature was multiplied by 1000 to create mixed numerical scales.",
        "All models used the same stratified 80/20 split (random_state=42).",
        "For scaled models, StandardScaler was inside a Pipeline and fitted on training data only.",
        "", "RESULTS TABLE", wide.round(4).to_string(), "",
        "OBSERVED EFFECT", effects[["Model", "Accuracy change (scaled - raw)", "F1 change (scaled - raw)", "Observed comparison"]].round(4).to_string(),
        "", "WHY SCALING DID OR DID NOT MATTER", ""
    ]
    for name, explanation in explanations.items():
        summary.extend([name + ":", explanation, ""])
    summary.extend([
        "Reporting rule: substantial means absolute accuracy or F1 change >= 0.02.",
        "Little/no measurable change means both absolute changes <= 0.005. These are practical cutoffs, not universal statistical rules.",
        "Results depend on the dataset, split, and model settings. Interpret the actual values rather than assuming scaling always helps.",
        "", "Files: results_comparison_table.csv, model_metrics_long.csv, scaling_effect_and_explanations.csv, accuracy_comparison.png, f1_comparison.png."
    ])
    (OUT / "SUMMARY.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"\nSaved tables, plots, and SUMMARY.txt in: {OUT.resolve()}")

if __name__ == "__main__":
    main()
1 