# Text Classification with Naive Bayes: Theory Meets Practice

## Dataset
Uses the SMS Spam Collection:
- `ham`: normal message
- `spam`: spam message

The script looks for `spam.csv` (label/message or v1/v2 columns), `SMSSpamCollection`, or `sms.tsv` beside the script. If none exists, it tries to download a public SMS TSV. If download fails, save the dataset as `sms.tsv` with two tab-separated columns: label and message.

## Run

```bash
python -m pip install numpy pandas matplotlib scikit-learn
python naive_bayes_project.py
```

The script creates `output/` automatically.

## Assignment coverage
1. **From-scratch GaussianNB:** class-wise means, variances, priors, normalized log-posteriors; prediction agreement compared with scikit-learn.
2. **Three variants:** GaussianNB uses dense TF-IDF features, MultinomialNB uses word counts, BernoulliNB uses binary presence/absence.
3. **Smoothing tuning:** GridSearchCV tests alpha values with stratified 5-fold cross-validation and spam F1. Tuning uses training data only.
4. **Word interpretation:** smoothed log-likelihood ratios show words associated with spam (positive) or ham (negative).
5. **Independence assumption:** discusses dependent phrases and measures the effect of deliberately duplicating one spam-associated word feature.

## Output files
- `Q1_scratch_vs_sklearn.csv`: true labels, predictions and agreement per message.
- `Q2_three_variant_comparison.csv`: metrics for all three Naive Bayes variants.
- `Q3_alpha_tuning_summary.csv`: best alpha, cross-validation F1 and test metrics.
- `Q3_multinomialnb_alpha_cv.csv`, `Q3_bernoullinb_alpha_cv.csv`: CV scores per alpha.
- `Q4_word_log_likelihood_ratios.csv`: word counts and log-likelihood ratios.
- `Q4_top_words_by_class.csv`: top spam-associated and ham-associated words.
- `Q4_top_word_log_ratios.png`: bar chart of word associations.
- `Q5_duplicate_feature_effect.csv`: probability shifts per test message.
- `Q5_independence_effect_metrics.csv`: before/after performance metrics.
- `Q5_duplicate_feature_effect.png`: probability comparison plot.
- `all_questions_metrics.csv`: combined metrics.
- `SUMMARY.txt`: generated summary with actual measured values and output inventory.

## Notes
- The split is stratified (75% train / 25% test) with a fixed seed.
- Vectorizers are fitted only on training messages.
- GaussianNB uses a limited dense TF-IDF matrix to keep memory manageable.
- Alpha is selected with cross-validation on training data, then evaluated on the held-out test set.
- Log-likelihood ratios show associations, not causal explanations.
- The duplicated-feature experiment is a controlled example of dependence; whether performance worsens must be judged from the saved metrics.
