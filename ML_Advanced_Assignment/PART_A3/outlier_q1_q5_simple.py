"""Simple implementation of outlier coding questions Q1-Q5.
Install: pip install numpy pandas matplotlib scikit-learn scipy
Run: python outlier_q1_q5_simple.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, skew
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer

rng = np.random.default_rng(42)
OUT = Path("outlier_outputs")
OUT.mkdir(exist_ok=True)

# Q1: Compare Z-score, IQR, Isolation Forest and LOF on one dataset.
def q1():
    normal = rng.multivariate_normal([0, 0], [[1, .65], [.65, 1]], 350)
    X = np.vstack([normal, rng.uniform(4, 6, (18, 2)), rng.normal([0, 4], .25, (8, 2))])
    df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    z = ((df - df.mean()).abs() / df.std(ddof=0) > 3).any(axis=1).to_numpy()
    q1v, q3v = df.quantile(.25), df.quantile(.75)
    iqr = q3v - q1v
    iqr_flag = ((df < q1v - 1.5*iqr) | (df > q3v + 1.5*iqr)).any(axis=1).to_numpy()
    flags = {
        "Z-score": z,
        "IQR": iqr_flag,
        "Isolation Forest": IsolationForest(contamination=.06, random_state=42).fit_predict(df) == -1,
        "LOF": LocalOutlierFactor(n_neighbors=20, contamination=.06).fit_predict(df) == -1
    }
    pca = PCA(n_components=2).fit_transform(df)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, flag) in zip(axes.ravel(), flags.items()):
        ax.scatter(pca[~flag, 0], pca[~flag, 1], s=18, label="Not flagged")
        ax.scatter(pca[flag, 0], pca[flag, 1], s=35, marker="x", label="Flagged")
        ax.set(title=f"{name}: {flag.sum()} flagged", xlabel="PC1", ylabel="PC2")
        ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"q1_detector_comparison.png", dpi=150); plt.close(fig)
    pd.DataFrame(flags).to_csv(OUT/"q1_flags.csv", index=False)
    pd.DataFrame({"method": list(flags), "flagged_count": [int(v.sum()) for v in flags.values()],
                  "flagged_percent": [round(v.mean()*100, 2) for v in flags.values()]}).to_csv(OUT/"q1_summary.csv", index=False)
    print("Q1 counts:", {k: int(v.sum()) for k, v in flags.items()})

# Q2: Measure coefficient changes after adding extreme errors to 5% of y values.
def q2():
    X = rng.normal(size=(250, 3))
    y = 5 + X @ np.array([3, -2, 1.5]) + rng.normal(0, .8, 250)
    y_bad = y.copy(); idx = rng.choice(len(y), int(.05*len(y)), replace=False)
    y_bad[idx] += rng.choice([-1, 1], len(idx))*25
    models = {"OLS": LinearRegression(), "Ridge": Ridge(alpha=5),
              "Huber": HuberRegressor(epsilon=1.35, alpha=0, max_iter=2000)}
    rows = []
    for name, model in models.items():
        model.fit(X, y); clean = np.r_[model.intercept_, model.coef_]
        model.fit(X, y_bad); dirty = np.r_[model.intercept_, model.coef_]
        for j, label in enumerate(["intercept", "beta_1", "beta_2", "beta_3"]):
            rows.append([name, label, clean[j], dirty[j], abs(dirty[j]-clean[j])])
    result = pd.DataFrame(rows, columns=["model", "coefficient", "clean_data", "contaminated_data", "absolute_change"])
    result.to_csv(OUT/"q2_coefficient_changes.csv", index=False)
    summary = result[result.coefficient != "intercept"].groupby("model")["absolute_change"].mean().rename("mean_absolute_slope_change").reset_index()
    summary.to_csv(OUT/"q2_sensitivity_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, group in result[result.coefficient != "intercept"].groupby("model"):
        ax.plot(group.coefficient, group.absolute_change, marker="o", label=name)
    ax.set(title="Q2: Coefficient changes after contamination", xlabel="Coefficient", ylabel="Absolute change")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/"q2_coefficient_sensitivity.png", dpi=150); plt.close(fig)
    print("Q2 mean slope changes:\n", summary.to_string(index=False))

# Q3: Compare empirical and robust Mahalanobis distances.
def q3():
    core = rng.multivariate_normal([0, 0], [[1, .75], [.75, 1]], 300)
    X = np.vstack([core, rng.uniform(4.5, 7, (20, 2))])
    emp = EmpiricalCovariance().fit(X)
    robust = MinCovDet(random_state=42, support_fraction=.75).fit(X)
    cutoff = chi2.ppf(.975, df=2)
    d1, d2 = emp.mahalanobis(X), robust.mahalanobis(X)
    f1, f2 = d1 > cutoff, d2 > cutoff
    pd.DataFrame({"x1": X[:,0], "x2": X[:,1], "empirical_d2": d1, "robust_d2": d2,
                  "empirical_flag": f1, "robust_flag": f2}).to_csv(OUT/"q3_mahalanobis_pointwise.csv", index=False)
    pd.DataFrame({"method": ["Empirical covariance", "MinCovDet robust covariance"],
                  "flagged_count": [int(f1.sum()), int(f2.sum())], "cutoff": [cutoff, cutoff]}).to_csv(OUT/"q3_mahalanobis_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, flag, name in zip(axes, [f1, f2], ["Empirical", "MinCovDet robust"]):
        ax.scatter(X[~flag,0], X[~flag,1], s=18, label="Not flagged")
        ax.scatter(X[flag,0], X[flag,1], s=35, marker="x", label="Flagged")
        ax.set(title=f"{name}: {flag.sum()} flagged", xlabel="x1", ylabel="x2"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"q3_mahalanobis_comparison.png", dpi=150); plt.close(fig)
    print(f"Q3 counts: empirical={f1.sum()}, MinCovDet={f2.sum()}, both={(f1&f2).sum()}, empirical-only={(f1&~f2).sum()}, robust-only={(~f1&f2).sum()}")

# Q4: Learn capping limits and Isolation Forest on training data only.
def q4():
    n = 700
    X = pd.DataFrame({"income": rng.lognormal(10.5, .45, n), "age": rng.normal(42,12,n).clip(18,85),
                      "spend": rng.gamma(3,400,n)})
    idx = rng.choice(n, 20, replace=False); X.loc[idx,"income"] *= 8; X.loc[idx,"spend"] *= 10
    y = .00003*X.income + .02*X.age + .0004*X.spend + rng.normal(0,1,n)
    Xtr, Xte, _, _ = train_test_split(X, y, test_size=.25, random_state=42)
    low, high = Xtr.quantile(.01), Xtr.quantile(.99)  # fit limits on train only
    cap = lambda data: data.clip(lower=low, upper=high, axis="columns")
    train, test = cap(Xtr), cap(Xte)
    detector = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             IsolationForest(n_estimators=200, contamination=.04, random_state=42))
    detector.fit(train)  # Never fit the detector on test data.
    tf, vf = detector.predict(train) == -1, detector.predict(test) == -1
    pd.concat([train.assign(is_outlier=tf, split="train"),
               test.assign(is_outlier=vf, split="test")]).sort_index().to_csv(OUT/"q4_capped_data_and_flags.csv", index_label="original_index")
    pd.DataFrame({"feature": X.columns, "lower_cap_from_train": low.values,
                  "upper_cap_from_train": high.values}).to_csv(OUT/"q4_training_cap_limits.csv", index=False)
    print(f"Q4 flags: train={tf.sum()}, test={vf.sum()}; thresholds and detector fitted on training data only.")

# Q5: Compare IQR flags before and after Yeo-Johnson transformation.
def q5():
    x = rng.lognormal(1.4, .9, 500); x[:8] *= 12
    original = pd.Series(x, name="original_value")
    transformed = pd.Series(PowerTransformer(method="yeo-johnson", standardize=False)
                            .fit_transform(original.to_frame()).ravel(), name="transformed_value")
    def iqr_flags(s):
        q1, q3 = s.quantile(.25), s.quantile(.75); spread = q3-q1
        return (s < q1-1.5*spread) | (s > q3+1.5*spread)
    before, after = iqr_flags(original), iqr_flags(transformed)
    pd.DataFrame({"original_value": original, "transformed_value": transformed,
                  "flag_before": before, "flag_after": after}).to_csv(OUT/"q5_transform_pointwise.csv", index=False)
    pd.DataFrame({"stage": ["Before Yeo-Johnson", "After Yeo-Johnson"],
                  "skewness": [skew(original), skew(transformed)],
                  "flagged_count": [int(before.sum()), int(after.sum())]}).to_csv(OUT/"q5_transform_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(np.arange(len(x)), original, c=np.where(before, "red", "steelblue"), s=15)
    axes[0].set(title=f"Before: {before.sum()} IQR flags", xlabel="Observation", ylabel="Original value")
    axes[1].scatter(np.arange(len(x)), transformed, c=np.where(after, "red", "steelblue"), s=15)
    axes[1].set(title=f"After: {after.sum()} IQR flags", xlabel="Observation", ylabel="Transformed value")
    fig.tight_layout(); fig.savefig(OUT/"q5_transform_comparison.png", dpi=150); plt.close(fig)
    print(f"Q5 skewness: {skew(original):.3f} -> {skew(transformed):.3f}; IQR flags: {before.sum()} -> {after.sum()}")
    print(f"Q5 before-only={(before & ~after).sum()}, after-only={(~before & after).sum()}")

if __name__ == "__main__":
    q1(); q2(); q3(); q4(); q5()
    print(f"\nDone. Results saved in: {OUT.resolve()}")
