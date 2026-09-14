# Instance-Based & Tree-Based Models — Customer Churn

## Project Overview

This project applies three machine learning approaches to customer churn prediction using the IBM Telco Customer Churn dataset:

* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest

The objective is to understand how instance-based and tree-based models work, compare their performance on the same churn dataset, investigate decision-tree overfitting, and select a suitable model for production churn scoring.

---

## Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains approximately 7,000 customer records with numerical and categorical features.

The target variable used in this project is:

```text
Churn Label
```

The target is converted into a binary variable:

* `No` → `0`
* `Yes` → `1`

---

## Project Structure

```text
project/
│
├── raw data/
│   └── Telco_customer_churn(1).csv
│
├── notebook/
│   └── churn_models.ipynb
│
├── results/
│   ├── model_comparison.csv
│   ├── rf_top5_importances.csv
│   └── part_c_model_selection_report.md
│
└── README.md
```

---

# Data Preparation

## Removing Data Leakage

Several columns were removed because they either identify the customer, contain location information that is not useful for general churn modeling, or are directly related to the churn outcome.

Removed columns include:

```text
CustomerID
Count
Country
State
City
Zip Code
Lat Long
Latitude
Longitude
Churn Label
Churn Value
Churn Score
Churn Reason
```

`Churn Label` is removed from the feature matrix because it is the target variable.

`Churn Value`, `Churn Score`, and `Churn Reason` were also removed because they are derived from or directly related to the churn outcome and could cause target leakage.

---

## Handling Total Charges

The `Total Charges` column was converted from text to numeric:

```python
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)
```

Rows with missing `Total Charges` values correspond to customers with zero tenure. These values were therefore set to zero:

```python
df.loc[
    df["Total Charges"].isna() &
    (df["Tenure Months"] == 0),
    "Total Charges"
] = 0
```

---

## Train/Test Split

The dataset was divided into training and testing sets using a stratified 80/20 split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Stratification was used to maintain a similar churn/non-churn proportion in both datasets.

---

# Part A — Concepts

## A1 — Why does a single Decision Tree have high variance?

A single decision tree can have high variance because small changes in the training data can result in different splits and therefore a very different tree structure.

A common symptom is very high training performance but noticeably lower test performance.

For example, a tree might achieve approximately 99% training accuracy but only 78% test accuracy, indicating possible overfitting.

---

## A2 — Why can weighted KNN outperform unweighted KNN?

In standard KNN, every neighbor has the same voting power.

Weighted KNN gives closer neighbors more influence and farther neighbors less influence.

This can improve predictions because nearby observations are often more similar to the new observation. The advantage can become more noticeable when using a larger value of `k`, because more distant observations are included.

---

## A3 — Why does Random Forest generalize better than a single deep tree?

Random Forest builds many decision trees using different bootstrap samples and random subsets of features.

The trees are therefore less correlated with each other. Their predictions are then combined, which reduces variance.

As a result, Random Forest generally generalizes better than a single deep decision tree.

---

## A4 — Why does KNN require feature scaling?

KNN makes predictions using distances between observations.

If features have very different scales, large-valued features can dominate the distance calculation.

For example:

* Tenure may range from 0–72 months.
* Total Charges may range from 0 to several thousand dollars.

Without scaling, Total Charges could have much more influence on the distance than Tenure.

Therefore, numerical features were standardized using `StandardScaler`.

---

## A5 — How does a Decision Tree choose a split?

A Decision Tree evaluates possible feature splits and selects a split that provides the greatest reduction in impurity, such as Gini impurity.

If the tree is allowed to grow without constraints, it can continue splitting until the training observations are almost perfectly separated.

This can produce very high training accuracy but poor performance on unseen data because the tree has overfit the training data.

---

# Part B — Experiments

## B1 — Random Forest

A Random Forest classifier was trained using:

```python
RandomForestClassifier(
    n_estimators=300,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
```

The model achieved an OOB score of approximately:

```text
0.8001
```

or:

```text
80.01%
```

### Top 5 Feature Importances

| Feature                   | Importance |
| ------------------------- | ---------: |
| Total Charges             |     0.1280 |
| Tenure Months             |     0.1164 |
| Monthly Charges           |     0.1080 |
| CLTV                      |     0.1050 |
| Contract — Month-to-month |     0.0509 |

### Sanity Check

The results are plausible.

* **Total Charges** captures accumulated customer spending and relationship duration.
* **Tenure Months** represents how long the customer has remained with the company.
* **Monthly Charges** represents the customer's current monthly spending.
* **CLTV** represents customer value.
* **Month-to-month Contract** represents the customer's level of contractual commitment.

These features can contain useful information for predicting churn.

Feature importance indicates predictive usefulness and should not be interpreted as proof that a feature causes churn.

---

# B2 — KNN Distance Metric Experiment

KNN was evaluated using two distance metrics:

* Euclidean distance
* Manhattan distance

The same preprocessing and `k = 11` were used for both models.

### Results

| Distance Metric | F1 Score |
| --------------- | -------: |
| Euclidean       |   0.5924 |
| Manhattan       |   0.5818 |

Euclidean distance performed better in this experiment.

The difference was approximately:

```text
0.0106
```

or about:

```text
1.06 percentage points
```

This result only indicates that Euclidean distance performed better under the current preprocessing and value of `k`. It does not mean Euclidean distance is always better than Manhattan distance.

---

# B3 — Head-to-Head Model Comparison

The three models were compared using the same training and testing data:

1. KNN
2. Decision Tree
3. Random Forest

The evaluation metrics were:

* Precision
* Recall
* F1
* PR-AUC
* Training time
* Inference time

The final results are stored in:

```text
results/model_comparison.csv
```

The comparison table generated by the notebook provides the evidence used for the final model-selection decision.

---

# B4 — Decision Tree Constraints

Two Decision Tree models were compared:

### Unconstrained Tree

```python
DecisionTreeClassifier(
    random_state=42
)
```

This tree is allowed to grow without explicit depth or leaf-size restrictions.

### Constrained Tree

```python
DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=20,
    random_state=42
)
```

The constrained tree limits the complexity of the model.

The comparison between training and testing performance demonstrates how an unconstrained tree can overfit.

The top levels of the constrained tree were also visualized to show how the model makes its initial splitting decisions.

---

# Part C — Model Selection

The final objective is to select a model for production customer churn scoring.

The three models are evaluated using:

* Precision
* Recall
* F1
* PR-AUC
* Training time
* Inference time

The final comparison is saved in:

```text
results/model_comparison.csv
```

---

## Production Model Recommendation

The final model should be selected using both predictive performance and practical production considerations.

Important factors include:

* F1 and PR-AUC
* Precision and recall
* Inference latency
* Training/retraining cost
* Interpretability
* Model complexity

The model with the strongest overall balance between predictive performance and production requirements should be recommended rather than selecting a model based only on accuracy.

The final recommendation is documented in the Part C model-selection report.

---

## Random Forest Feature Importance

The five most important Random Forest features were:

1. Total Charges
2. Tenure Months
3. Monthly Charges
4. CLTV
5. Contract — Month-to-month

These features are reasonable predictors of churn because they describe customer spending, relationship duration, customer value, and contractual commitment.

The feature importance results are also saved in:

```text
results/rf_top5_importances.csv
```

---

# Production Monitoring

One concrete monitoring check for the selected production model is the **predicted churn rate over time**.

For example, the percentage of customers predicted as churners can be monitored weekly or monthly.

If the predicted churn rate changes substantially from its normal range, this could indicate:

* Data drift
* Changes in customer behavior
* Changes in the customer population
* A preprocessing or pipeline problem

A significant change should trigger an investigation and potentially a model retraining process.

---

# Conclusion

This project compares an instance-based model, KNN, with tree-based models, Decision Tree and Random Forest.

KNN requires feature scaling because it relies on distance calculations. The Decision Tree is easy to interpret but can overfit when allowed to grow without constraints. Random Forest reduces variance by combining many diverse decision trees.

The final production recommendation is based on the head-to-head model comparison from Part B while also considering practical factors such as interpretability, inference latency, and retraining cost.

All final experimental results are stored in the `results/` directory.
