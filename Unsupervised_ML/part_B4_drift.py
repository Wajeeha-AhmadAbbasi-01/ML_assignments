import pandas as pd
import joblib
import json
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split


# --------------------------------
# 1. Load trained model
# --------------------------------

best_pipeline = joblib.load("churn_pipeline.joblib")


# --------------------------------
# 2. Load and clean dataset
# --------------------------------

df = pd.read_csv("data/Telco_customer_churn copy.csv")

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

df.loc[
    df["Total Charges"].isnull() & (df["Tenure Months"] == 0),
    "Total Charges"
] = 0


# --------------------------------
# 3. Select model features
# --------------------------------

features = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Contract"
]

X = df[features].copy()

y = df["Churn Label"].map({
    "Yes": 1,
    "No": 0
})


# --------------------------------
# 4. Recreate the same test split
# --------------------------------

_, X_test, _, _ = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# 5. Simulate data drift
# --------------------------------

X_test_drifted = X_test.copy()

X_test_drifted["Monthly Charges"] += 20


# --------------------------------
# 6. KS Test
# --------------------------------

ks_statistic, p_value = ks_2samp(
    X_test["Monthly Charges"],
    X_test_drifted["Monthly Charges"]
)

print("KS Test")
print("--------------------")
print(f"KS statistic: {ks_statistic:.4f}")
print(f"p-value: {p_value:.4f}")


# --------------------------------
# 7. Predictions before drift
# --------------------------------

preds_before = best_pipeline.predict_proba(
    X_test
)[:, 1]


# --------------------------------
# 8. Predictions after drift
# --------------------------------

preds_after = best_pipeline.predict_proba(
    X_test_drifted
)[:, 1]


mean_before = preds_before.mean()
mean_after = preds_after.mean()


print("\nPrediction Impact")
print("--------------------")

print(
    f"Mean churn probability before: "
    f"{mean_before:.4f}"
)

print(
    f"Mean churn probability after: "
    f"{mean_after:.4f}"
)


# --------------------------------
# 9. Determine drift status
# --------------------------------

if p_value < 0.05:
    drift_detected = True
else:
    drift_detected = False


# --------------------------------
# 10. Save results
# --------------------------------

drift_results = {
    "feature_tested": "Monthly Charges",
    "simulated_shift": "+20",
    "ks_statistic": float(ks_statistic),
    "p_value": float(p_value),
    "drift_detected": drift_detected,
    "mean_churn_probability_before": float(mean_before),
    "mean_churn_probability_after": float(mean_after)
}

with open("drift_results.json", "w") as file:
    json.dump(drift_results, file, indent=4)


print("\nResults saved to drift_results.json")