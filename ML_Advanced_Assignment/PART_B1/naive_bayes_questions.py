"""
Naive Bayes Applied Questions (Q1-Q5)
Dataset: SMS Spam Collection (ham/spam messages)

Run:
    python naive_bayes_project.py

The script creates an output folder containing each question's results and plots.
It uses the same SMS dataset for all five questions. If internet access is unavailable,
place the dataset CSV/TSV in the same folder as this script (see README.md).
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


OUTPUT_DIR = Path("output")
RANDOM_STATE = 42
MAX_FEATURES = 300


def load_sms_dataset():
    """Load a local SMS dataset, or download a public copy."""
    possible_files = [Path("spam.csv"), Path("SMSSpamCollection"), Path("sms.tsv")]
    for path in possible_files:
        if path.exists():
            if path.name == "SMSSpamCollection":
                data = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], encoding="latin-1")
            elif path.suffix.lower() == ".tsv":
                data = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], encoding="latin-1")
            else:
                data = pd.read_csv(path, encoding="latin-1")
                label_col = next((c for c in data.columns if c.lower() in ["v1", "label", "category"]), None)
                message_col = next((c for c in data.columns if c.lower() in ["v2", "message", "text"]), None)
                if label_col is None or message_col is None:
                    raise ValueError("spam.csv must contain label and message/text columns (or v1 and v2).")
                data = data[[label_col, message_col]].rename(columns={label_col: "label", message_col: "message"})
            break
    else:
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        try:
            data = pd.read_csv(url, sep="\t", header=None, names=["label", "message"], encoding="latin-1")
        except Exception as exc:
            raise RuntimeError(
                "Could not download the SMS dataset. Download a public SMS Spam Collection file "
                "and save it beside this script as sms.tsv (two tab-separated columns: label, message)."
            ) from exc

    data = data[["label", "message"]].dropna()
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data = data[data["label"].isin(["ham", "spam"])].copy()
    data["target"] = (data["label"] == "spam").astype(int)
    return data.reset_index(drop=True)


def scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_spam": precision_score(y_true, y_pred, zero_division=0),
        "recall_spam": recall_score(y_true, y_pred, zero_division=0),
        "f1_spam": f1_score(y_true, y_pred, zero_division=0),
    }


class ScratchGaussianNB:
    """Small Gaussian Naive Bayes implementation for dense numeric features."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.means_ = []
        self.variances_ = []
        self.priors_ = []

        # Match sklearn GaussianNB's global variance smoothing.
        epsilon = 1e-9 * np.var(X, axis=0).max()
        self.epsilon_ = epsilon

        for class_value in self.classes_:
            X_class = X[y == class_value]
            self.means_.append(X_class.mean(axis=0))
            self.variances_.append(X_class.var(axis=0) + epsilon)
            self.priors_.append(len(X_class) / len(X))

        self.means_ = np.asarray(self.means_)
        self.variances_ = np.asarray(self.variances_)
        self.priors_ = np.asarray(self.priors_)
        return self

    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_posts = []
        for i in range(len(self.classes_)):
            mean = self.means_[i]
            var = self.variances_[i]
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * var) + ((X - mean) ** 2) / var,
                axis=1,
            )
            log_posts.append(np.log(self.priors_[i]) + log_likelihood)

        log_posts = np.column_stack(log_posts)
        # Normalize log-posteriors using log-sum-exp.
        max_log = log_posts.max(axis=1, keepdims=True)
        return log_posts - max_log - np.log(np.exp(log_posts - max_log).sum(axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_sms_dataset()
    X_text = data["message"].astype(str)
    y = data["target"].to_numpy()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Shared feature representations; vectorizers are fitted on training text only.
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
    X_test_tfidf = tfidf.transform(X_test_text).toarray()

    counts = CountVectorizer(max_features=MAX_FEATURES, stop_words="english")
    X_train_count = counts.fit_transform(X_train_text)
    X_test_count = counts.transform(X_test_text)

    results = []
    summaries = [
        "Naive Bayes Applied Questions — Results Summary",
        f"Dataset: SMS messages ({len(data)} rows; spam rate: {y.mean():.2%})",
        "All questions use the same SMS Spam Collection dataset and a fixed random seed.",
        "",
    ]

    # Q1: Implement GaussianNB from scratch and compare with sklearn.
    scratch = ScratchGaussianNB().fit(X_train_tfidf, y_train)
    sklearn_gnb = GaussianNB().fit(X_train_tfidf, y_train)
    scratch_pred = scratch.predict(X_test_tfidf)
    sklearn_pred = sklearn_gnb.predict(X_test_tfidf)
    agreement = np.mean(scratch_pred == sklearn_pred)
    q1_scores = scores(y_test, scratch_pred)
    pd.DataFrame({
        "actual": y_test,
        "scratch_prediction": scratch_pred,
        "sklearn_prediction": sklearn_pred,
    }).to_csv(OUTPUT_DIR / "Q1_gaussian_nb_predictions.csv", index=False)
    results.append({"question": "Q1 Scratch GaussianNB", **q1_scores, "agreement_with_sklearn": agreement})
    summaries += [
        "Q1 — Gaussian Naive Bayes from scratch",
        f"Prediction agreement with sklearn GaussianNB: {agreement:.4f}",
        f"Scratch model test accuracy: {q1_scores['accuracy']:.4f}",
        "The implementation estimates each class-feature mean and variance, then combines Gaussian log-likelihoods with class priors.",
        "",
    ]

    # Q2: Grid-search alpha for MultinomialNB using count features.
    alpha_rows = []
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train_count, y_train)
        pred = model.predict(X_test_count)
        alpha_rows.append({"alpha": alpha, **scores(y_test, pred)})
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(OUTPUT_DIR / "Q2_alpha_grid_search.csv", index=False)
    best_row = alpha_df.loc[alpha_df["f1_spam"].idxmax()]
    plt.figure(figsize=(7, 4))
    plt.plot(alpha_df["alpha"], alpha_df["f1_spam"], marker="o", label="F1 for spam")
    plt.plot(alpha_df["alpha"], alpha_df["precision_spam"], marker="s", label="Precision for spam")
    plt.plot(alpha_df["alpha"], alpha_df["recall_spam"], marker="^", label="Recall for spam")
    plt.xscale("log")
    plt.xlabel("Smoothing parameter alpha (log scale)")
    plt.ylabel("Validation score")
    plt.title("Q2: Effect of alpha on MultinomialNB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Q2_alpha_grid_search.png", dpi=150)
    plt.close()
    results.append({"question": "Q2 MultinomialNB alpha search", **best_row.to_dict()})
    summaries += [
        "Q2 — MultinomialNB smoothing parameter",
        f"Best alpha by spam F1 on the held-out validation set: {best_row['alpha']}",
        f"At that alpha: accuracy={best_row['accuracy']:.4f}, precision={best_row['precision_spam']:.4f}, recall={best_row['recall_spam']:.4f}, F1={best_row['f1_spam']:.4f}.",
        "Alpha adds smoothing to word counts. The CSV and plot show how the scores change across tested alpha values.",
        "",
    ]

    # Q3: BernoulliNB uses binary presence; MultinomialNB uses word counts.
    binary_vectorizer = CountVectorizer(max_features=MAX_FEATURES, stop_words="english", binary=True)
    X_train_binary = binary_vectorizer.fit_transform(X_train_text)
    X_test_binary = binary_vectorizer.transform(X_test_text)
    bernoulli = BernoulliNB(alpha=1.0).fit(X_train_binary, y_train)
    multinomial = MultinomialNB(alpha=1.0).fit(X_train_count, y_train)
    q3_rows = [
        {"model": "BernoulliNB (binary presence)", **scores(y_test, bernoulli.predict(X_test_binary))},
        {"model": "MultinomialNB (word counts)", **scores(y_test, multinomial.predict(X_test_count))},
    ]
    pd.DataFrame(q3_rows).to_csv(OUTPUT_DIR / "Q3_bernoulli_vs_multinomial.csv", index=False)
    results.extend([{"question": "Q3 " + row["model"], **{k: v for k, v in row.items() if k != "model"}} for row in q3_rows])
    best_q3 = max(q3_rows, key=lambda row: row["f1_spam"])
    summaries += [
        "Q3 — BernoulliNB vs MultinomialNB",
        f"Better spam F1 on this split: {best_q3['model']} (F1={best_q3['f1_spam']:.4f}).",
        "BernoulliNB models whether a word appears at least once; MultinomialNB also uses how many times it appears. Results depend on the corpus and split.",
        "",
    ]

    # Q4: Duplicate one feature and observe posterior probability changes.
    # This deliberately violates Naive Bayes' conditional-independence assumption.
    original_model = MultinomialNB(alpha=1.0).fit(X_train_count, y_train)
    duplicate_train = np.hstack([X_train_count.toarray(), X_train_count[:, :1].toarray()])
    duplicate_test = np.hstack([X_test_count.toarray(), X_test_count[:, :1].toarray()])
    duplicate_model = MultinomialNB(alpha=1.0).fit(duplicate_train, y_train)
    original_prob = original_model.predict_proba(X_test_count)[:, 1]
    duplicate_prob = duplicate_model.predict_proba(duplicate_test)[:, 1]
    q4_df = pd.DataFrame({
        "actual": y_test,
        "spam_probability_original": original_prob,
        "spam_probability_after_duplicate": duplicate_prob,
        "absolute_probability_shift": np.abs(duplicate_prob - original_prob),
    })
    q4_df.to_csv(OUTPUT_DIR / "Q4_duplicate_feature_probabilities.csv", index=False)
    mean_shift = q4_df["absolute_probability_shift"].mean()
    changed = np.mean((original_prob >= 0.5) != (duplicate_prob >= 0.5))
    plt.figure(figsize=(6, 5))
    plt.scatter(original_prob, duplicate_prob, alpha=0.45)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Spam probability before duplication")
    plt.ylabel("Spam probability after duplication")
    plt.title("Q4: Effect of duplicating a word-count feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Q4_duplicate_feature_probabilities.png", dpi=150)
    plt.close()
    results.append({
        "question": "Q4 Duplicate correlated feature",
        "mean_absolute_probability_shift": mean_shift,
        "fraction_predictions_changed_at_0.5": changed,
    })
    summaries += [
        "Q4 — Double counting from a duplicated feature",
        f"Mean absolute shift in predicted spam probability: {mean_shift:.4f}",
        f"Fraction of predictions that changed at threshold 0.5: {changed:.2%}",
        "The same word-count feature was added twice. Naive Bayes treats the duplicate as separate evidence, so probabilities can become more extreme or shift incorrectly.",
        "",
    ]

    # Q5: Learning curves for GaussianNB and Logistic Regression on the same TF-IDF features.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    train_sizes = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    curve_rows = []
    plt.figure(figsize=(7, 4.5))
    for name, estimator in [
        ("GaussianNB", GaussianNB()),
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ]:
        sizes, train_scores, valid_scores = learning_curve(
            estimator,
            X_train_tfidf,
            y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )
        train_mean = train_scores.mean(axis=1)
        valid_mean = valid_scores.mean(axis=1)
        plt.plot(sizes, valid_mean, marker="o", label=f"{name} validation")
        for size, tr, va in zip(sizes, train_mean, valid_mean):
            curve_rows.append({
                "model": name,
                "training_rows": size,
                "mean_training_accuracy": tr,
                "mean_validation_accuracy": va,
            })
    plt.xlabel("Number of training messages")
    plt.ylabel("Accuracy")
    plt.title("Q5: Learning curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Q5_learning_curves.png", dpi=150)
    plt.close()
    pd.DataFrame(curve_rows).to_csv(OUTPUT_DIR / "Q5_learning_curve_scores.csv", index=False)
    curve_df = pd.DataFrame(curve_rows)
    summaries += [
        "Q5 — Learning curves: GaussianNB vs Logistic Regression",
        "The plot shows cross-validated accuracy as the number of training messages increases.",
        "GaussianNB is generative: it models each class's feature distributions. Logistic Regression is discriminative: it learns the class boundary directly.",
        "Compare the curves rather than assuming one model always learns faster; the outcome depends on the dataset and feature representation.",
        "",
    ]

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "all_questions_metrics.csv", index=False)
    summaries += [
        "Files created",
        "- Q1_gaussian_nb_predictions.csv",
        "- Q2_alpha_grid_search.csv and Q2_alpha_grid_search.png",
        "- Q3_bernoulli_vs_multinomial.csv",
        "- Q4_duplicate_feature_probabilities.csv and Q4_duplicate_feature_probabilities.png",
        "- Q5_learning_curve_scores.csv and Q5_learning_curves.png",
        "- all_questions_metrics.csv",
        "- SUMMARY.txt",
        "",
    ]
    (OUTPUT_DIR / "SUMMARY.txt").write_text("\n".join(summaries), encoding="utf-8")
    print("\n".join(summaries))
    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
