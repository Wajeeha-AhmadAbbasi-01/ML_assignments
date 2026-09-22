# Part C — End-to-End ML Engineering & Deployment

## 1. Customer Segmentation and Business Implications

For the unsupervised learning component, KMeans clustering was applied to the standardized numerical features:

- Tenure Months
- Monthly Charges
- Total Charges
- CLTV

The silhouette-score analysis tested values of `k` from 2 to 8. The highest silhouette score was obtained at **k = 2**, with a score of **0.3936**.

The two resulting clusters were profiled using their average customer characteristics and observed churn rates.

| Cluster | Avg. Tenure | Avg. Monthly Charges | Avg. Total Charges | Avg. CLTV | Observed Churn Rate |
|---|---:|---:|---:|---:|---:|
| 0 | 19.54 | 53.33 | 899.91 | 4054.80 | 31.76% |
| 1 | 57.98 | 87.78 | 5056.13 | 5092.21 | 16.24% |

Cluster 0 has shorter average tenure, lower total charges, lower CLTV, and a higher observed churn rate. This suggests that it is a higher-churn segment that may require greater retention attention, particularly around onboarding, service experience, and early customer engagement. Cluster 1 has longer tenure, higher accumulated charges, higher CLTV, and a lower observed churn rate. The business could therefore focus on maintaining satisfaction and loyalty within this more established customer segment.

**Important:** Churn was not used as an input to KMeans. The churn rates above were calculated after clustering to understand how the resulting segments relate to observed churn.

---

## 2. PCA Dimensionality Reduction

PCA was applied to the four standardized numerical features.

### Explained Variance

| Principal Components | Cumulative Explained Variance |
|---|---:|
| 1 | 59.41% |
| 2 | 83.16% |
| 3 | 98.53% |
| 4 | 100.00% |

Three principal components were retained for the PCA experiment.

- Original feature shape: **(5634, 4)**
- PCA feature shape: **(5634, 3)**
- Variance retained by 3 components: **98.53%**
- PCA + KMeans silhouette score: **0.3435**

This shows that three principal components provide substantial dimensionality reduction while retaining most of the information represented by the original four numerical features.

### PCA Result :

> Explained variance ratio:
[0.59405875 0.23751959 0.15376528 0.01465638]

Cumulative explained variance:
[0.59405875 0.83157834 0.98534362 1.        ]
1 components: 0.5941
2 components: 0.8316
3 components: 0.9853
4 components: 1.0000

Original shape: (5634, 4)
PCA shape: (5634, 3)
PCA + KMeans silhouette: 0.3435448454204931
---

## 3. Silhouette-Based KMeans Selection

KMeans was evaluated for `k = 2` through `k = 8` using silhouette score.

| k | Silhouette Score |
|---:|---:|
| 2 | **0.3936** |
| 3 | 0.3352 |
| 4 | 0.3387 |
| 5 | 0.3601 |
| 6 | 0.3524 |
| 7 | 0.3467 |
| 8 | 0.3217 |

The highest silhouette score among the tested values was **0.3936 at k = 2**. Therefore, `k = 2` was used for the final customer segmentation analysis.

### Silhouette Plot

> **Attach screenshot here**
>
> The plot is availale in notebook/Part_B.ipynb

---

## 4. Reproducible Churn Prediction Pipeline

A complete supervised ML pipeline was created for churn prediction.

The pipeline contains:

1. Numerical preprocessing using `StandardScaler`
2. Categorical preprocessing using `OneHotEncoder(handle_unknown="ignore")`
3. Logistic Regression classifier
4. Hyperparameter tuning using `GridSearchCV`
5. Cross-validation using 5 folds
6. Model selection using F1 score

The model uses the following features:

- Tenure Months
- Monthly Charges
- Total Charges
- Contract

The target variable is `Churn`, encoded as:

- `Yes = 1`
- `No = 0`

The complete fitted pipeline was saved as:

`churn_pipeline.joblib`

This ensures that preprocessing and prediction steps remain together when the model is deployed.

### Model Results

Moddel results are shown in `model_metrics.json`:

---

## 5. Pipeline Save and Reload Verification

The saved pipeline was reloaded using `joblib.load()` and used to generate predictions on the same test data.

The original and reloaded model probabilities were compared.


## 6. FastAPI Deployment

The trained churn pipeline was packaged behind a FastAPI application.

The API provides two main endpoints:

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Churn Prediction

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

### FastAPI Swagger Screenshot

> **Attach FastAPI screenshot here**
> ![FastAPI Request](Unsupervised_ML/prediction input.jpg)
> ![Prediction Response](Unsupervised_ML/prediction response.jpg)
  

## 7. Simulated Production Drift Monitoring

A simple drift experiment was performed by creating a modified copy of the test data and increasing `Monthly Charges` by **20**.

The Kolmogorov-Smirnov (KS) test was used to compare the original and modified `Monthly Charges` distributions.

The model's average churn probability was also compared before and after the simulated shift.

### Drift Results

Drift results are in  `drift_results.json`:


A commonly used threshold for this experiment is `p < 0.05`. A p-value below this threshold provides evidence that the two feature distributions differ.

**Note:** This is a simulated drift experiment. It does not prove that real production data is currently experiencing drift.


## 8. Production Monitoring Plan

One concrete monitoring check is to monitor the **distribution of input features**, such as `Monthly Charges`, over time.

If the production distribution changes significantly compared with the training/reference distribution, this may indicate data drift. The model's prediction distribution can also be monitored to identify unusual changes in predicted churn probabilities.

Other useful production checks include:

- API latency
- API error rate
- Missing or invalid input values
- Prediction distribution
- Feature distribution drift
- Model performance when actual churn outcomes become available

If substantial drift or performance degradation is detected, the pipeline should be investigated and retraining can be considered.

---

## 9. Production Readiness

The project demonstrates an end-to-end workflow from data preparation and unsupervised segmentation to supervised churn prediction and API deployment.

The main production components are:

- Reproducible preprocessing and model pipeline
- Hyperparameter tuning
- Saved model artifact
- Pipeline reload verification
- FastAPI prediction endpoint
- Health-check endpoint
- Input validation through Pydantic
- Simulated drift detection
- Prediction monitoring plan

For a production system, additional improvements would be needed, such as:

- Automated monitoring dashboards
- Automated drift alerts
- A defined retraining schedule
- Model performance monitoring once real outcomes are available
- Model/version tracking
- Logging and error monitoring
- Rollback to a previous model version
- Authentication and access control for the API

---

## Part C Summary

This project combines customer segmentation and deployable churn prediction in one end-to-end workflow. PCA was used to reduce the numerical feature space while retaining 98.53% of the variance with three components. KMeans clustering was evaluated using silhouette scores, with `k = 2` producing the highest score among the tested values. The resulting customer segments showed different customer characteristics and observed churn rates, providing a basis for targeted business analysis. A reproducible Logistic Regression pipeline was then created for churn prediction, saved with Joblib, and verified after reloading. Finally, the pipeline was deployed through FastAPI and a simulated feature-distribution shift was used to demonstrate a basic production drift-monitoring approach.
