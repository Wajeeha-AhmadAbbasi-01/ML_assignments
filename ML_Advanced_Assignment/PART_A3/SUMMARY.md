# Outlier Detection and Treatment — Summary (Q1–Q5)

## Project goal
This project demonstrates five practical outlier detection and treatment tasks using Python and synthetic data. The script creates example data with unusual observations, runs each experiment, and saves plots and CSV results.

## Setup and run
Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

Run the script from its folder:

```bash
python outlier_q1_q5_simple.py
```

Results are saved to the `outlier_outputs/` folder.

## Q1 — Compare four outlier detectors
**Performed:** Created a 2D dataset with a main cluster, distant points, and a smaller separate group. Applied Z-score, IQR, Isolation Forest, and Local Outlier Factor (LOF). PCA was used to plot all four sets of flags in a common 2D view.

**Why:** Z-score and IQR use feature-wise thresholds. Isolation Forest identifies points that are easy to isolate, while LOF looks for unusually low local density.

**Outputs:** `q1_detector_comparison.png`, `q1_flags.csv`, `q1_summary.csv`.

**Interpretation:** Different methods can flag different points. The largest flagged count is not necessarily the most accurate result; ground-truth labels are needed to assess detection accuracy.

## Q2 — OLS, Ridge, and Huber sensitivity
**Performed:** Generated a regression dataset, fitted OLS, Ridge, and Huber regression, added large errors to 5% of target values, and refitted the models. Absolute coefficient changes were calculated and plotted.

**Why:** OLS minimizes squared errors and can be affected by large residuals. Ridge penalizes coefficient size. Huber regression reduces the influence of large residuals.

**Outputs:** `q2_coefficient_sensitivity.png`, `q2_coefficient_changes.csv`, `q2_sensitivity_summary.csv`.

**Interpretation:** Smaller coefficient changes mean greater stability in this experiment. The outliers were added to the target variable `y`, so this primarily tests vertical outliers. Ridge regularization is not the same as robust regression.

## Q3 — Empirical vs. robust Mahalanobis distance
**Performed:** Created correlated 2D data with distant observations. Computed squared Mahalanobis distances using empirical covariance and robust Minimum Covariance Determinant (MinCovDet). Used the 97.5th percentile of a chi-square distribution with two degrees of freedom as the illustrative cutoff.

**Why:** Mahalanobis distance accounts for feature correlation. Extreme points can distort empirical covariance; MinCovDet tries to estimate the central data distribution robustly.

**Outputs:** `q3_mahalanobis_comparison.png`, `q3_mahalanobis_pointwise.csv`, `q3_mahalanobis_summary.csv`.

**Interpretation:** Compare counts and point-by-point flags. The cutoff assumes an approximately multivariate-normal central distribution and is not a universal rule. More flags do not automatically mean better detection.

## Q4 — Train-only Isolation Forest and capping
**Performed:** Split data first. Learned 1st and 99th percentile capping limits from training features only, applied the same limits to train and test, fitted an imputation/scaling/Isolation Forest pipeline on training data, then used that fitted pipeline to flag both sets.

**Why:** Fitting capping thresholds or an anomaly detector on the full dataset would allow test data to influence training (data leakage).

**Outputs:** `q4_capped_data_and_flags.csv`, `q4_training_cap_limits.csv`.

**Interpretation:** The test set is processed with training-derived limits and the same fitted detector. The example keeps all rows, caps feature values, and retains flags for review. In a real predictive project, evaluate downstream model performance on an untouched test set.

## Q5 — Yeo-Johnson and IQR flags
**Performed:** Generated a strongly right-skewed feature, flagged outliers using IQR, applied Yeo-Johnson, and applied IQR again. Compared skewness, flagged counts, and which observations changed flag status.

**Why:** A power transformation changes feature spacing and quartiles and may reduce skewness. Yeo-Johnson can handle zero and negative values, unlike Box-Cox.

**Outputs:** `q5_transform_comparison.png`, `q5_transform_pointwise.csv`, `q5_transform_summary.csv`.

**Interpretation:** Fewer flags do not prove that genuine anomalies disappeared; no observations are removed by the transformation itself. For predictive workflows, fit the transformer on training data only and apply it to test data using `.transform()`.

## Overall conclusions
1. Different outlier methods use different definitions and can flag different observations.
2. Huber regression is designed to reduce the influence of large residuals; Ridge mainly controls coefficient size.
3. Robust covariance may help when outliers distort ordinary covariance estimates.
4. Fit thresholds, transformers, and detectors on training data only, then reuse them on test data.
5. Transformations can change outlier flags because they change the distribution and IQR fences.

## Output files
All plots and CSV result tables are written to `outlier_outputs/`. Numerical results should be taken from the output of your own run because they depend on the generated data, random seed, and method settings.
