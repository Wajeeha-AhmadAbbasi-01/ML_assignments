# IBM Telco Customer Churn — Unsupervised Learning & ML Engineering

## Project Overview

This project applies unsupervised learning and machine learning engineering techniques to the IBM Telco Customer Churn dataset.

The assignment combines:

1. Customer segmentation using PCA and KMeans.
2. Reproducible supervised churn prediction.
3. FastAPI deployment.
4. Basic production drift monitoring.

The overall workflow goes from data preparation and segmentation to model training, saving, deployment, and monitoring.

---

## Dataset

Dataset:

```text
Telco_customer_churn.csv
```

The dataset contains approximately 7,000 customer records with numerical and categorical information.

The supervised target is:

```text
Churn
```

where:

- `Yes` = customer churned
- `No` = customer did not churn

For clustering, the churn label is **not used as an input feature**. It is only used afterward to profile the resulting clusters.

---

# Part A — Concepts and ML Engineering

## A1. KMeans vs DBSCAN

### KMeans

KMeans is a partition-based clustering algorithm. It requires the number of clusters `k` to be selected and assigns observations to the nearest centroid. It generally works well when clusters are relatively compact and roughly spherical.

### DBSCAN

DBSCAN is a density-based clustering algorithm. It does not require the number of clusters to be specified beforehand, can discover irregularly shaped clusters, and can identify low-density observations as noise or outliers.

For customer segmentation, DBSCAN's noise detection can be useful for identifying unusual customer behavior.

---

## A2. Feature Store / Shared Feature Computation

A feature store or shared feature-computation library helps ensure that the same feature definitions are used during training and prediction.

Without shared feature logic, training and production code may calculate the same feature differently, causing training-serving skew.

Benefits include:

- Consistent feature definitions
- Reusable transformations
- Less duplicated code
- Better reproducibility
- Reduced training-serving differences

---

## A3. Preventing Data Leakage

Preprocessing should be fitted only on the training portion of each cross-validation fold.

For example, `StandardScaler` should learn its mean and standard deviation from the training fold and then transform the validation fold.

A scikit-learn `Pipeline` helps enforce this process and prevents information from the validation data leaking into preprocessing.

---

## A4. Production Monitoring

A production ML system should monitor:

- Input feature distributions
- Feature drift
- Prediction distribution
- API latency
- API errors
- Missing or invalid inputs
- Model performance when actual outcomes become available

A significant change in feature distributions may indicate data drift. A change in the relationship between inputs and actual churn outcomes may indicate concept drift.

---

## A5. DBSCAN Noise Detection

DBSCAN can identify low-density observations as noise. For customer segmentation, this can help identify atypical customers whose behavior does not fit the main customer groups.

---

# Part B — Implementation

## B1. PCA Dimensionality Reduction

PCA was applied to:

- Tenure Months
- Monthly Charges
- Total Charges
- CLTV

The numerical features were standardized before PCA.

### Explained Variance

| Component | Explained Variance |
|---|---:|
| PC1 | 59.41% |
| PC2 | 23.75% |
| PC3 | 15.38% |
| PC4 | 1.47% |

### Cumulative Explained Variance

| Components | Cumulative Variance |
|---:|---:|
| 1 | 59.41% |
| 2 | 83.16% |
| 3 | 98.53% |
| 4 | 100.00% |

Three components were retained.

```text
Original shape: (5634, 4)
PCA shape:      (5634, 3)
```

The three components retained approximately **98.53% of the original variance**.

PCA + KMeans produced a silhouette score of:

```text
0.3435
```

---

## B2. KMeans Selection Using Silhouette Score

KMeans was tested for `k = 2` through `k = 8`.

| k | Silhouette Score |
|---:|---:|
| 2 | **0.3936** |
| 3 | 0.3352 |
| 4 | 0.3387 |
| 5 | 0.3601 |
| 6 | 0.3524 |
| 7 | 0.3467 |
| 8 | 0.3217 |

The highest tested silhouette score was:

```text
Best k = 2
Best silhouette score = 0.3936
```

Therefore, `k = 2` was used for the final segmentation.

---

## B3. Reproducible Churn Prediction Pipeline

A separate supervised pipeline was created for churn prediction.

### Input Features

```text
Tenure Months
Monthly Charges
Total Charges
Contract
```

### Target

```text
Churn
```

Mapping:

```text
Yes → 1
No  → 0
```

### Preprocessing

Numerical features:

```text
StandardScaler
```

Categorical features:

```text
OneHotEncoder(handle_unknown="ignore")
```

### Model

```text
Logistic Regression
```

Hyperparameter tuning was performed using `GridSearchCV` with:

```text
C = [0.01, 0.1, 1, 10]
5-fold cross-validation
scoring = F1
```

The complete fitted pipeline was saved as:

```text
churn_pipeline.joblib
```

---

## B3.1 FastAPI Deployment

The saved pipeline was deployed through FastAPI.

### Health Endpoint

```text
GET /health
```

Example:

```json
{
  "status": "ok"
}
```

### Prediction Endpoint

```text
POST /predict
```

Example request:

```json
{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 840.0,
  "Contract": "Month-to-month"
}
```

Example response:

```json
{
  "churn_probability": 0.63
}
```

The exact probability depends on the trained model.



# B4 — Production Drift Simulation

A simulated feature shift was created by increasing `Monthly Charges` by:

```text
+20
```

The original and shifted distributions were compared using the Kolmogorov-Smirnov test.

The model's mean churn probability was also compared before and after the shift.

### Drift Results

The exact values are stored in:

```text
drift_results.json
```

```text
p < 0.05
```

A p-value below this threshold provides evidence that the two distributions differ.

**Important:** this is a simulated experiment and does not prove that real production data currently has drift.


# Part C — End-to-End ML Engineering

## C1. Customer Segmentation and Business Implications

The final KMeans model used:

```text
k = 2
```

Cluster profiling produced:

| Cluster | Avg. Tenure | Avg. Monthly Charges | Avg. Total Charges | Avg. CLTV | Observed Churn |
|---|---:|---:|---:|---:|---:|
| 0 | 19.54 | 53.33 | 899.91 | 4054.80 | 31.76% |
| 1 | 57.98 | 87.78 | 5056.13 | 5092.21 | 16.24% |

Cluster 0 has shorter average tenure, lower total charges, lower CLTV, and a higher observed churn rate. This suggests that it is a higher-churn segment that may require greater retention attention, particularly around onboarding, service experience, and early customer engagement.

Cluster 1 has longer tenure, higher accumulated charges, higher CLTV, and a lower observed churn rate. The business could focus on maintaining satisfaction and loyalty within this more established customer segment.

**Important:** Churn was not used to create the KMeans clusters. It was only used afterward to profile the segments.

---

## C2. Pipeline Save and Reload Verification

The complete fitted pipeline was saved with Joblib and then reloaded.

The original and reloaded predictions were compared.

Expected result:

```text
Predictions identical: True
```

This verifies that the saved pipeline can be reloaded and produces the same predictions.

---

## C3. Model Evaluation

The supervised model was evaluated using:

- Accuracy
- Precision
- Recall
- F1
- PR-AUC

The exact values are stored in:

```text
model_metrics.json
```
F1 combines precision and recall, while PR-AUC is useful when evaluating churn prediction with class imbalance.

---

## C4. Production Monitoring

One concrete monitoring check is to monitor the distribution of important input features over time, such as `Monthly Charges`.

Production data can be compared with the training/reference distribution.

A significant change may indicate:

- Data drift
- Changes in customer behavior
- A data-pipeline problem
- A change in the population being scored

The prediction distribution should also be monitored.

Additional checks include:

- API latency
- API error rate
- Missing values
- Invalid inputs
- Prediction distribution
- Feature drift
- Model performance when actual outcomes become available

---

## C5. Production Readiness

The project demonstrates an end-to-end ML workflow:

```text
Raw Dataset
    ↓
Data Cleaning
    ↓
Feature Preparation
    ↓
Standardization
    ↓
PCA / KMeans Segmentation
    ↓
Customer Segment Analysis

Raw Dataset
    ↓
Train/Test Split
    ↓
Preprocessing Pipeline
    ↓
Logistic Regression
    ↓
GridSearchCV
    ↓
Saved Pipeline
    ↓
FastAPI
    ↓
Prediction
    ↓
Monitoring
```

For a production system, further improvements would include:

- Automated monitoring dashboards
- Automated drift alerts
- Model versioning
- Defined retraining schedule
- Model performance monitoring
- Logging
- API authentication
- Access control
- Rollback to previous model versions
- CI/CD testing and deployment

---


---

# How to Run

## Install Dependencies

```bash
pip install pandas numpy scikit-learn scipy matplotlib fastapi uvicorn joblib
```

## Train the Model

```bash
python train_model.py
```

This creates:

```text
churn_pipeline.joblib
model_metrics.json
```

## Run Drift Simulation

```bash
python b4_drift.py
```

This creates:

```text
drift_results.json
```

## Start FastAPI

From the project folder:

```bash
uvicorn app:app --reload
```

The API is normally available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Use Swagger to test:

```text
GET /health
POST /predict
```

---

# Example API Test

### Request

```json
{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 840.0,
  "Contract": "Month-to-month"
}
```

### Response

```json
{
  "churn_probability": 0.63
}
```

The returned value is an example; the actual probability depends on the trained model.

---

# Key Learning Outcomes

## Unsupervised Learning

- KMeans clustering
- DBSCAN concepts
- Silhouette score
- Customer segmentation
- PCA
- Explained variance
- Dimensionality reduction

## Supervised Machine Learning

- Logistic Regression
- Standardization
- One-hot encoding
- Cross-validation
- GridSearchCV
- Precision
- Recall
- F1
- PR-AUC

## ML Engineering

- Reproducible pipelines
- Joblib model serialization
- Save/reload verification
- FastAPI deployment
- Pydantic validation
- API health checks
- Drift detection
- Production monitoring

---


# Conclusion

This project demonstrates an end-to-end machine learning workflow using the IBM Telco Customer Churn dataset.

The unsupervised component used PCA and KMeans to identify customer segments without using the churn label as a clustering input. Silhouette analysis identified `k = 2` as the highest-scoring option among the tested values. The resulting clusters showed different customer characteristics and observed churn rates, demonstrating how segmentation can support targeted business analysis.

The supervised component created a reproducible Logistic Regression pipeline containing preprocessing and prediction. The pipeline was tuned using cross-validation, saved with Joblib, reloaded successfully, and exposed through a FastAPI application.

Finally, a simulated feature shift demonstrated a basic approach to production drift monitoring.

Overall, the assignment connects **unsupervised learning, supervised prediction, reproducibility, API deployment, and production monitoring** into one end-to-end ML engineering workflow.
