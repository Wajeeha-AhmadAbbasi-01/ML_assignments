# EHR Missing Data Project — Summary

**File:** `ehr_missing_data_project.py`

**Topic:** Diagnosing and Repairing Missingness in EHR Data (Part A1 — Handling Missing Data)

---

## 1. What This Project Does

This project demonstrates a complete, end-to-end workflow for diagnosing and repairing missingness in Electronic Health Record (EHR) data. It uses the **Pima Indians Diabetes dataset** and simulates known missingness mechanisms (MCAR, MAR, MNAR) so that every design decision can be traced back to a known ground truth.

The single Python script performs the following 22 stages:

| # | Stage | Description |
|---|-------|-------------|
| 1 | Imports | Loads all required libraries (numpy, pandas, sklearn, matplotlib, seaborn, joblib) |
| 2 | Settings | Sets random seed (42), test size (0.20), CV folds (5), and creates the `outputs/` folder |
| 3 | Load Dataset | Downloads the Pima Indians Diabetes dataset (768 rows × 9 columns) |
| 4 | Impossible Zeros | Converts clinically impossible zeros to `NaN` for Glucose, BloodPressure, SkinThickness, Insulin, BMI (zero is meaningful for Pregnancies, so it is kept) |
| 5 | Feature/Target Split | Separates `X_base` (features) from `y` (Outcome) |
| 6 | Reference Data | Creates a median-filled reference copy **only** for controlled simulation (never used as a pre-fitted imputer for the final model) |
| 7 | Inject MCAR/MAR/MNAR | - **MCAR:** SkinThickness missing at ~15%, independent of all values<br>- **MAR:** BMI missingness depends on observed **Age** (logistic, capped at 30%)<br>- **MNAR:** Insulin missingness depends on the **true Insulin** value (logistic, capped at 35%) |
| 8 | Missingness Profile | Computes missing count and missing rate per feature; saves a bar chart |
| 9 | Missingness Heatmap | Produces a missingness-pattern heatmap across all observations |
| 10 | Missingness Correlation | Computes correlation between missingness indicators; saves a correlation heatmap |
| 11 | Missingness vs Target | Compares target (diabetes) rate when a feature is missing vs observed; saves a grouped bar chart |
| 12 | Mechanism Diagnosis | Documents the known injected mechanisms and chosen repairs |
| 13 | Feature-Specific Pipeline | Builds a leakage-free `ColumnTransformer` with:<br>- **MICE / IterativeImputer** → BMI (MAR)<br>- **Median + missing indicator** → Insulin (MNAR)<br>- **Median imputation** → all other features |
| 14 | Train/Test Evaluation | Fits the full pipeline on the training set; reports ROC-AUC, Accuracy, Precision, Recall, F1; saves classification report and confusion matrix |
| 15 | Cross-Validation | Runs 5-fold StratifiedKFold CV on the full pipeline using ROC-AUC |
| 16 | Ablation Study | Compares 4 imputation strategies (Mean, Median, Median+Indicator, MICE) using the same CV folds |
| 17 | Ablation Plot | Bar chart of mean 5-fold ROC-AUC for each strategy |
| 18 | Save Pipeline | Saves the final fitted pipeline as `21_ehr_missingness_final_pipeline.joblib` |
| 19 | Simulation Info | Saves a table describing the simulation mechanisms |
| 20 | Project Summary | Writes a plain-text summary with all final metrics |
| 21 | Strategy Table | Saves a per-feature diagnosis / strategy / justification table |
| 22 | List Outputs | Prints every file generated in the `outputs/` folder |

---

## 2. Key Design Decisions

- **Leakage prevention:** All imputation and preprocessing steps live inside `sklearn` `Pipeline` / `ColumnTransformer` objects, so they are re-fitted **inside every CV training fold**.
- **Feature-specific repair:** Each missingness mechanism gets an imputation strategy that matches its statistical properties:
  - MCAR → simple median imputation is sufficient.
  - MAR → MICE uses the relationships among observed variables.
  - MNAR → median imputation **plus** a missingness indicator preserves the information that the value was missing.
- **Simulation ground truth:** Because MCAR/MAR/MNAR are deliberately injected, the "correct" repair strategy is known — this makes the ablation study meaningful.

---

## 3. Outputs Stored in the `outputs/` Folder

| # | File | Description |
|---|------|-------------|
| 01 | `01_original_pima_data.csv` | Raw Pima dataset as downloaded |
| 02 | `02_ehr_with_baseline_missing_values.csv` | Dataset after converting impossible zeros to NaN |
| 03 | `03_simulated_ehr_dataset.csv` | Final dataset with injected MCAR/MAR/MNAR missingness |
| 04 | `04_missingness_profile.csv` | Missing count and missing rate per feature |
| 05 | `05_missingness_rate.png` | Bar chart of missingness rate by feature |
| 06 | `06_missingness_pattern_heatmap.png` | Heatmap of missingness patterns |
| 07 | `07_missingness_correlation.csv` | Correlation matrix of missingness indicators |
| 08 | `08_missingness_correlation_heatmap.png` | Heatmap of missingness correlations |
| 09 | `09_missingness_vs_target.csv` | Target rate when missing vs observed |
| 10 | `10_missingness_vs_target.png` | Grouped bar chart of target rate |
| 11 | `11_missingness_diagnosis.csv` | Injected mechanism and chosen repair per feature |
| 12 | `12_final_test_metrics.csv` | Final test-set metrics (AUC, Acc, Prec, Rec, F1) |
| 13 | `13_classification_report.csv` | Full sklearn classification report |
| 14 | `14_confusion_matrix.csv` | Confusion matrix as CSV |
| 15 | `15_confusion_matrix.png` | Confusion matrix plot |
| 16 | `16_final_pipeline_cv_scores.csv` | Per-fold ROC-AUC for the final pipeline |
| 17 | `17_final_pipeline_cv_summary.csv` | Mean and std CV ROC-AUC |
| 18 | `18_ablation_results.csv` | Mean/std AUC per ablation strategy |
| 19 | `19_detailed_ablation_results.csv` | Per-fold AUC per ablation strategy |
| 20 | `20_ablation_auc_comparison.png` | Bar chart comparing ablation strategies |
| 21 | `21_ehr_missingness_final_pipeline.joblib` | Saved final pipeline (ready for inference) |
| 22 | `22_simulation_information.csv` | Simulation mechanism definitions and rate controls |
| 23 | `23_project_summary.txt` | Plain-text summary of the entire project |
| 24 | `24_feature_strategy_table.csv` | Per-feature diagnosis / strategy / justification |

---

## 4. Final Metrics (as produced by the script)

- **Final Test ROC-AUC:** printed at runtime (≈0.79–0.82 on Pima)
- **Final Test Accuracy / Precision / Recall / F1:** saved in `12_final_test_metrics.csv`
- **Final Pipeline Mean CV ROC-AUC:** saved in `17_final_pipeline_cv_summary.csv`
- **Best Ablation Strategy:** the strategy with the highest mean 5-fold ROC-AUC (saved in `18_ablation_results.csv`)

---

## 5. Important Limitation

The missingness mechanisms are **known** only because MCAR, MAR, and MNAR were deliberately injected for this project. In a real EHR dataset, MAR and especially MNAR generally **cannot be proven from observed data alone** — this is why the simulation-based design of this project is a teaching/validation tool rather than a claim about real-world diagnosis.

---

## 6. Leakage Prevention

Every imputation and preprocessing step is contained inside `sklearn` `Pipeline` / `ColumnTransformer` objects. During cross-validation, the pipeline is re-fitted on each training fold only, so no information from the validation fold leaks into the imputation or scaling steps.

---

## 7. How to Run

```bash
python ehr_missing_data_project.py