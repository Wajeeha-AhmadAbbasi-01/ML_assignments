# PART A2 — Fraud Detection Under Extreme Imbalance

## 1. Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning under extreme class imbalance.

Fraudulent transactions represent only a very small percentage of all transactions. Consequently, a model can achieve high accuracy while failing to detect fraudulent transactions.

The purpose of this assignment is to explore imbalance-handling techniques, perform leakage-safe model evaluation, optimize decision thresholds, and select a model based on business cost rather than accuracy alone.

## 2. Project Objectives

The main objectives are:

- Analyze the class distribution and feature patterns in the fraud dataset.
- Understand why accuracy is misleading for highly imbalanced classification.
- Implement Stratified K-Fold cross-validation.
- Compare baseline classification, class weighting, SMOTE, and random undersampling.
- Evaluate models using Precision, Recall, and PR-AUC.
- Prevent data leakage by applying preprocessing and resampling inside cross-validation folds.
- Optimize the classification threshold according to a business cost function.
- Implement SMOTE interpolation from scratch on a toy dataset.
- Compare ensemble-based imbalance-handling approaches.
- Generate comparison tables, visualizations, and a final recommendation.

## 3. Dataset

**Dataset:** Credit Card Fraud Detection

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The commonly distributed version contains 284,807 transactions, including 492 fraudulent transactions.

| Property | Description |
|---|---|
| Target column | `Class` |
| Class 0 | Legitimate transaction |
| Class 1 | Fraudulent transaction |
| Total transactions | 284,807 in the original dataset |
| Fraud transactions | 492 in the original dataset |
| Fraud rate | Approximately 0.172% |

The dataset contains anonymized numerical features (`V1`–`V28`), transaction time (`Time`), transaction amount (`Amount`), and the target variable (`Class`).

The actual dataset counts should be verified by running the project.

## 4. Folder Structure

The project is organized into separate folders for the complete project and individual questions.

```text
PART_A2/
│
├── output/
│   │
│   ├── PROJECT/
│   │   └── Complete fraud detection project outputs
│   │
│   ├── Q1/
│   │   └── Stratified K-Fold + SMOTE Pipeline
│   │
│   ├── Q2/
│   │   └── Comparison of imbalance-handling strategies
│   │
│   ├── Q3/
│   │   └── Decision threshold optimization
│   │
│   ├── Q4/
│   │   └── SMOTE implementation from scratch
│   │
│   ├── Q5/
│   │   └── Ensemble-based imbalance comparison
│   │
│   ├── dataset_class_distribution.csv
│   ├── final_summary.csv
│   └── final_summary.txt
│
├── creditcard.csv
├── fraud_detection_project.py
├── questions.py
└── README.md
```

The folders inside `output/` store the generated results for the complete project and individual questions.

## 5. Description of Questions

### Q1 — Stratified K-Fold with SMOTE

**Objective:** Evaluate a fraud detection model using Stratified K-Fold cross-validation with SMOTE inside an `imblearn` Pipeline.

The model is trained using resampled training folds, while validation folds remain untouched.

**Key concepts:**
- Stratified cross-validation
- SMOTE
- Pipeline-based preprocessing
- PR-AUC
- Data leakage prevention

### Q2 — Comparison of Imbalance-Handling Strategies

**Objective:** Compare four approaches using the same base classifier and cross-validation folds.

The strategies are:

1. Baseline Logistic Regression
2. Logistic Regression with `class_weight="balanced"`
3. SMOTE with Logistic Regression
4. Random undersampling with Logistic Regression

**Evaluation metrics:**
- Precision
- Recall
- F1-score, where included
- PR-AUC
- Business cost, where included

The objective is to understand how different imbalance-handling techniques affect fraud detection.

### Q3 — Decision Threshold Optimization

**Objective:** Find a probability threshold that improves fraud detection while controlling false positives.

Instead of relying exclusively on the default threshold of 0.50, the model's predicted probabilities are evaluated at different thresholds.

The question also considers a target recall, such as 90%, where required.

The results include the selected threshold and confusion matrix.

### Q4 — SMOTE from Scratch

**Objective:** Understand how SMOTE generates synthetic minority-class observations.

A two-dimensional toy dataset is used to demonstrate the interpolation process between minority-class observations.

The implementation and visualization help explain how synthetic samples are generated and why SMOTE can help with class imbalance.

### Q5 — Ensemble Methods for Imbalanced Classification

**Objective:** Compare an ensemble method designed for imbalanced data with a Random Forest approach using SMOTE.

Depending on the implementation, the comparison includes:

- Balanced Random Forest
- Random Forest with SMOTE
- Identical cross-validation folds
- PR-AUC and other classification metrics

The purpose is to investigate whether an ensemble-based method provides better fraud detection performance than resampling with a conventional Random Forest.

## 6. Complete Project Methodology

The `PROJECT` folder contains the end-to-end fraud detection workflow.

### Step 1: Exploratory Data Analysis

Analyze class distribution, missing values, and selected feature distributions.

**Purpose:** Understand the dataset and the severity of class imbalance before model training.

### Step 2: Data Preparation

Separate the features (`X`) from the target (`y`) and prepare the training and test data.

The test set is reserved for final evaluation.

### Step 3: Cross-Validation

Use `StratifiedKFold` to maintain approximately the same class proportions across folds.

Use the same folds when comparing strategies to make the evaluation fairer.

### Step 4: Imbalance Handling

Evaluate baseline Logistic Regression, class weighting, SMOTE, and random undersampling.

Resampling is applied only to the training portion of each fold to avoid leakage.

### Step 5: Evaluation

Use PR-AUC, Precision, Recall, and confusion matrices to understand model performance.

Accuracy is not treated as the primary metric because fraud is extremely rare.

### Step 6: Business Cost Optimization

The project assumes that a false negative costs 20 times as much as a false positive.

The cost function is:

`Expected Cost = 20 × FN + FP`

Where:

- `FN` = fraudulent transactions incorrectly classified as legitimate.
- `FP` = legitimate transactions incorrectly classified as fraudulent.

For example, 5 false negatives and 100 false positives result in:

`Expected Cost = (20 × 5) + 100 = 200`

The cost values are relative units based on the assignment's assumptions, not measured financial losses.

### Step 7: Threshold Selection

Use out-of-fold predictions to evaluate different classification thresholds and select a threshold that minimizes the expected cost.

Keep the final test set separate from threshold optimization.

### Step 8: Final Recommendation

Compare the evaluated strategies and recommend an approach based on expected business cost, supported by PR-AUC, Precision, Recall, and confusion matrix results.

## 7. Technologies and Libraries

The project uses:

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- imbalanced-learn
- Jupyter Notebook, if using the notebook workflow

Install the required packages with:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
```

## 8. How to Run the Project

### Step 1: Verify the dataset

Ensure that `creditcard.csv` is located in the expected directory and contains the `Class` target column.

### Step 2: Run the complete project

From the `PART_A2` directory, run:

```bash
python fraud_detection_project.py
```

### Step 3: Run the individual questions

The individual question implementations are organized in `questions.py` and the `output/Q1` through `output/Q5` folders.

Check the code in `questions.py` to identify how each question is selected or executed.

### Step 4: Review the generated outputs

Open the relevant folders under `output/` to review the question results, CSV files, visualizations, and summary reports.

**Important:** Confirm that the script loads the real `creditcard.csv` instead of generating a synthetic fallback dataset. Clearly identify synthetic results if they are used for testing.

## 9. Main Output Files

| File or folder | Purpose |
|---|---|
| `output/PROJECT/` | Results from the complete fraud detection workflow. |
| `output/Q1/` | Results for SMOTE and Stratified K-Fold evaluation. |
| `output/Q2/` | Results comparing imbalance-handling strategies. |
| `output/Q3/` | Threshold optimization results and confusion matrices. |
| `output/Q4/` | SMOTE-from-scratch implementation and visualizations. |
| `output/Q5/` | Ensemble-method comparison results. |
| `output/dataset_class_distribution.csv` | Summary of the dataset's class distribution. |
| `output/final_summary.csv` | Structured summary of the results. |
| `output/final_summary.txt` | Human-readable summary of the results. |

## 10. Expected Learning Outcomes

After completing this assignment, we should understand:

- Why extreme class imbalance creates challenges for classification.
- Why accuracy alone is insufficient for fraud detection.
- How Stratified K-Fold cross-validation works.
- How data leakage occurs during resampling.
- How class weighting, SMOTE, and undersampling differ.
- How Precision, Recall, and PR-AUC evaluate rare positive classes.
- How a probability threshold affects false positives and false negatives.
- How ensemble methods can be used for imbalanced classification.
- How business costs can guide model and threshold selection.

## 11. Conclusion

Part A2 demonstrates a practical machine learning workflow for fraud detection under extreme class imbalance.

The assignment combines exploratory data analysis, leakage-safe cross-validation, resampling techniques, threshold optimization, and ensemble learning.

The final goal is not simply to maximize accuracy. It is to identify a reliable fraud detection strategy that performs well on rare fraud cases and minimizes expected business cost under the assumption that false negatives are 20 times more expensive than false positives.

The final recommendation should be based on the actual experimental results obtained from the dataset.
