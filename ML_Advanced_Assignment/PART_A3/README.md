# Outlier Detection and Treatment — UCI Air Quality

## Dataset (original, real measurements)
This project uses the **UCI Air Quality** dataset: 9,358 hourly observations from a gas multisensor deployed in a polluted Italian city (March 2004–February 2005). The dataset contains real sensor responses and reference-analyzer pollutant concentrations.

- Official dataset page: https://archive.ics.uci.edu/dataset/360/air+quality
- Download `AirQualityUCI.csv` from the UCI page.
- Put it next to `outlier_project.ipynb`, or in a folder named `data/`.
- UCI represents missing sensor measurements with `-200`; the code converts these to missing values.
- Dataset is for research use; commercial use is excluded by the UCI terms.

## Run it
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook outlier_project.ipynb
```
Or run the same workflow as a Python script:
```bash
python outlier_project.py
```

## What the project does
1. Compares IQR, Isolation Forest, and Local Outlier Factor on the same chronological training period.
2. Saves the individual flags and overlap counts, so the methods' flagged sets can be compared.
3. Compares three regression pipelines predicting `CO(GT)`:
   - no outlier treatment (median imputation only),
   - cap predictor values at the training 1st/99th percentiles,
   - Yeo–Johnson transform predictors.
4. Uses an 80/20 chronological split to avoid training on future observations.
5. Saves performance metrics and plots under `outputs/`.

## Important interpretation / rare-signal risk
**Do not automatically delete flagged rows.** This is an environmental dataset, and a very high pollutant concentration may be a real pollution episode rather than bad data. The target `CO(GT)` is never capped or removed by the workflow. Predictor capping limits are learned from the training period only and then applied unchanged to the test period. Compare MAE/RMSE/R² before deciding whether capping helps. A better metric alone is not enough to prove a reading is noise; inspect timestamps, related pollutant sensors, sensor documentation, and physical plausibility.

## Feature-by-feature treatment
`outputs/feature_treatment_decisions_template.csv` gives a starter recommendation for each predictor. Review it alongside the distributions and domain knowledge. Weather features (`T`, `RH`, `AH`) are initially retained, since their extremes may be meaningful. Pollution and sensor-response features are tested under no treatment, capping, and transformation rather than automatically deleted.

## Outputs
- `detector_summary.csv` — number/percentage flagged by each detector.
- `detector_flags_train.csv` — row-level flags for each detector.
- `detector_overlap_counts.csv` — pairwise overlap counts between flagged sets.
- `model_performance_comparison.csv` — held-out MAE, RMSE, and R².
- `feature_treatment_decisions_template.csv` — per-feature rationale template.
- PNG plots — detector counts and predictions.

## Source citation
Vito, S. (2008). *Air Quality*. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F.
