# Part C — Model Selection Report

## Objective

The objective is to select a suitable model for production customer churn scoring by comparing KNN, Decision Tree, and Random Forest.

## Model Comparison

| Model | Precision | Recall | F1 | PR-AUC | Training Time (s) | Inference Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| KNN | 0.5810 | 0.6043 | 0.5924 | 0.5828 | 0.0338 | 0.0322 |
| Decision Tree | 0.6708 | 0.4358 | 0.5284 | 0.5789 | 0.0399 | 0.0103 |
| Random Forest | 0.6412 | 0.5160 | 0.5719 | 0.6420 | 0.8949 | 0.0804 |

## Recommended Model

The model with the highest F1 score in the head-to-head comparison is **KNN**.

The recommendation should not be based only on raw accuracy. For production churn scoring, factors such as model performance, inference latency, interpretability, and retraining cost should also be considered.

## Random Forest Top 5 Feature Importances

1. Total Charges — 0.1280
2. Tenure Months — 0.1164
3. Monthly Charges — 0.1080
4. CLTV — 0.1050
5. Contract_Month-to-month — 0.0509

These features are plausible predictors of churn. Tenure and contract type describe the customer's relationship with the company, while monthly and total charges describe customer spending. CLTV represents customer value.

Feature importance indicates predictive usefulness and does not imply causation.

## Production Monitoring

One concrete monitoring check would be to monitor the **predicted churn rate over time**.

If the percentage of customers predicted to churn changes significantly compared with the normal range, this could indicate data drift, a change in customer behavior, or a problem in the prediction pipeline. The model should then be investigated and potentially retrained.
