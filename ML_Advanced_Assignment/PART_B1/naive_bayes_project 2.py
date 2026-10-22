from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

OUT = Path("output")
SEED = 42
MAX_FEATURES = 500

def load_dataset():
    for path in [Path("spam.csv"), Path("SMSSpamCollection"), Path("sms.tsv")]:
        if not path.exists():
            continue
        if path.name == "SMSSpamCollection" or path.suffix.lower() == ".tsv":
            df = pd.read_csv(path, sep="\\t", header=None, names=["label", "message"], encoding="latin-1")
        else:
            raw = pd.read_csv(path, encoding="latin-1")
            lc = next((c for c in raw.columns if c.lower() in ["v1", "label", "category"]), None)
            mc = next((c for c in raw.columns if c.lower() in ["v2", "message", "text"]), None)
            if lc is None or mc is None:
                raise ValueError("spam.csv must contain label/message columns or v1/v2.")
            df = raw[[lc, mc]].rename(columns={lc: "label", mc: "message"})
        break
    else:
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        try:
            df = pd.read_csv(url, sep="\\t", header=None, names=["label", "message"], encoding="latin-1")
        except Exception as exc:
            raise RuntimeError("Dataset download failed. Save the SMS dataset beside this script as sms.tsv.") from exc
    df = df[["label", "message"]].dropna().copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["target"] = (df["label"] == "spam").astype(int)
    return df.reset_index(drop=True)

def metrics(y, pred):
    return {"accuracy": accuracy_score(y, pred),
            "spam_precision": precision_score(y, pred, zero_division=0),
            "spam_recall": recall_score(y, pred, zero_division=0),
            "spam_f1": f1_score(y, pred, zero_division=0)}

class ScratchGaussianNB:
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.epsilon_ = 1e-9 * np.var(X, axis=0).max()
        self.means_, self.vars_, self.priors_ = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            self.means_.append(Xc.mean(axis=0))
            self.vars_.append(Xc.var(axis=0) + self.epsilon_)
            self.priors_.append(len(Xc) / len(X))
        self.means_, self.vars_, self.priors_ = map(np.asarray, (self.means_, self.vars_, self.priors_))
        return self

    def predict_log_proba(self, X):
        X = np.asarray(X, float)
        scores = []
        for i in range(len(self.classes_)):
            var, mean = self.vars_[i], self.means_[i]
            ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mean) ** 2 / var, axis=1)
            scores.append(np.log(self.priors_[i]) + ll)
        scores = np.column_stack(scores)
        m = scores.max(axis=1, keepdims=True)
        return scores - (m + np.log(np.exp(scores - m).sum(axis=1, keepdims=True)))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

def main():
    OUT.mkdir(exist_ok=True)
    df = load_dataset()
    Xtext, y = df["message"].astype(str), df["target"].to_numpy()
    Xtr_text, Xte_text, ytr, yte = train_test_split(
        Xtext, y, test_size=0.25, random_state=SEED, stratify=y
    )
    summary = [ "TEXT CLASSIFICATION WITH NAIVE BAYES — SUMMARY",
                f"Rows: {len(df)}", f"Spam rate: {y.mean():.2%}",
                "Split: stratified 75% train / 25% test; random_state=42", "" ]
    all_rows = []

    count_vec = CountVectorizer(max_features=MAX_FEATURES, stop_words="english")
    Xtr_count, Xte_count = count_vec.fit_transform(Xtr_text), count_vec.transform(Xte_text)
    binary_vec = CountVectorizer(max_features=MAX_FEATURES, stop_words="english", binary=True)
    Xtr_bin, Xte_bin = binary_vec.fit_transform(Xtr_text), binary_vec.transform(Xte_text)
    tfidf_vec = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
    Xtr_tfidf = tfidf_vec.fit_transform(Xtr_text).toarray()
    Xte_tfidf = tfidf_vec.transform(Xte_text).toarray()

    # Q1: GaussianNB from scratch versus scikit-learn
    scratch = ScratchGaussianNB().fit(Xtr_tfidf, ytr)
    library = GaussianNB().fit(Xtr_tfidf, ytr)
    p1, p2 = scratch.predict(Xte_tfidf), library.predict(Xte_tfidf)
    agreement = float(np.mean(p1 == p2))
    pd.DataFrame({"actual": yte, "scratch_prediction": p1, "sklearn_prediction": p2,
                  "match": p1 == p2}).to_csv(OUT / "Q1_scratch_vs_sklearn.csv", index=False)
    s = metrics(yte, p1)
    all_rows.append({"question": "Q1 Scratch GaussianNB", **s, "agreement": agreement})
    summary += ["Q1 — Scratch GaussianNB vs sklearn", f"Prediction agreement: {agreement:.4f}",
                f"Scratch test accuracy: {s['accuracy']:.4f}",
                "The scratch model estimates class means, variances and priors, then computes normalized log-posteriors.", ""]

    # Q2: compare three NB variants with appropriate representations
    variants = [
        ("GaussianNB + TF-IDF", GaussianNB(), Xtr_tfidf, Xte_tfidf),
        ("MultinomialNB + counts", MultinomialNB(), Xtr_count, Xte_count),
        ("BernoulliNB + binary", BernoulliNB(), Xtr_bin, Xte_bin),
    ]
    rows = []
    for name, model, xtr, xte in variants:
        model.fit(xtr, ytr)
        row = {"model": name, **metrics(yte, model.predict(xte))}
        rows.append(row)
        all_rows.append({"question": "Q2 Model comparison", **row})
    pd.DataFrame(rows).to_csv(OUT / "Q2_three_variant_comparison.csv", index=False)
    summary += ["Q2 — Three Naive Bayes variants"]
    for r in rows:
        summary.append(f"{r['model']}: accuracy={r['accuracy']:.4f}, precision={r['spam_precision']:.4f}, recall={r['spam_recall']:.4f}, F1={r['spam_f1']:.4f}")
    summary.append("")

    # Q3: tune smoothing with 5-fold CV on training data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tuning = [
        ("MultinomialNB", MultinomialNB(), Xtr_count, Xte_count),
        ("BernoulliNB", BernoulliNB(), Xtr_bin, Xte_bin),
    ]
    tune_rows = []
    for name, model, xtr, xte in tuning:
        search = GridSearchCV(model, {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]},
                              scoring="f1", cv=cv, n_jobs=1, refit=True)
        search.fit(xtr, ytr)
        pred = search.best_estimator_.predict(xte)
        row = {"model": name, "best_alpha": search.best_params_["alpha"],
               "best_cv_spam_f1": search.best_score_, **metrics(yte, pred)}
        tune_rows.append(row)
        cvdf = pd.DataFrame(search.cv_results_)
        pd.DataFrame({"alpha": cvdf["param_alpha"].astype(float),
                      "mean_cv_f1": cvdf["mean_test_score"],
                      "std_cv_f1": cvdf["std_test_score"]}).to_csv(
                          OUT / f"Q3_{name.lower()}_alpha_cv.csv", index=False)
    pd.DataFrame(tune_rows).to_csv(OUT / "Q3_alpha_tuning_summary.csv", index=False)
    summary += ["Q3 — Alpha tuning with 5-fold cross-validation",
                "Alpha is selected using training data only; the test set is evaluated after selection."]
    for r in tune_rows:
        summary.append(f"{r['model']}: best alpha={r['best_alpha']}, best CV F1={r['best_cv_spam_f1']:.4f}, test F1={r['spam_f1']:.4f}")
    summary.append("")

    # Q4: smoothed word log-likelihood ratios, estimated from training data only
    word_names = count_vec.get_feature_names_out()
    train_counts = Xtr_count.toarray()
    spam_counts = train_counts[ytr == 1].sum(axis=0)
    ham_counts = train_counts[ytr == 0].sum(axis=0)
    alpha = 1.0
    p_spam = (spam_counts + alpha) / (spam_counts.sum() + alpha * len(word_names))
    p_ham = (ham_counts + alpha) / (ham_counts.sum() + alpha * len(word_names))
    ratios = np.log(p_spam / p_ham)
    word_df = pd.DataFrame({"word": word_names, "spam_count": spam_counts,
                            "ham_count": ham_counts, "log_ratio_spam_vs_ham": ratios})
    word_df = word_df.sort_values("log_ratio_spam_vs_ham", ascending=False)
    word_df.to_csv(OUT / "Q4_word_log_likelihood_ratios.csv", index=False)
    top_spam, top_ham = word_df.head(20), word_df.tail(20).sort_values("log_ratio_spam_vs_ham")
    pd.concat([top_spam.assign(class_interpretation="spam-associated"),
               top_ham.assign(class_interpretation="ham-associated")]).to_csv(
                   OUT / "Q4_top_words_by_class.csv", index=False)
    plot_df = pd.concat([top_ham.head(12), top_spam.head(12)]).sort_values("log_ratio_spam_vs_ham")
    plt.figure(figsize=(9, 7))
    plt.barh(plot_df["word"], plot_df["log_ratio_spam_vs_ham"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Log likelihood ratio (spam vs ham)")
    plt.title("Q4: Words associated with spam or ham")
    plt.tight_layout()
    plt.savefig(OUT / "Q4_top_word_log_ratios.png", dpi=150)
    plt.close()
    summary += ["Q4 — Most influential words by smoothed log-likelihood ratio",
                "Positive values indicate spam-associated words; negative values indicate ham-associated words.",
                "Top spam words: " + ", ".join(top_spam["word"].head(10).tolist()),
                "Top ham words: " + ", ".join(top_ham["word"].head(10).tolist()),
                "These are corpus-specific associations, not causal explanations.", ""]

    # Q5: natural phrase dependence plus a measurable duplicate-feature experiment
    baseline = MultinomialNB(alpha=1.0).fit(Xtr_count, ytr)
    base_prob = baseline.predict_proba(Xte_count)[:, 1]
    duplicated_word = str(top_spam.iloc[0]["word"])
    j = int(np.where(word_names == duplicated_word)[0][0])
    Xtr_dup = np.hstack([Xtr_count.toarray(), Xtr_count[:, j].toarray()])
    Xte_dup = np.hstack([Xte_count.toarray(), Xte_count[:, j].toarray()])
    duplicate = MultinomialNB(alpha=1.0).fit(Xtr_dup, ytr)
    dup_prob = duplicate.predict_proba(Xte_dup)[:, 1]
    shift = float(np.mean(np.abs(dup_prob - base_prob)))
    changed = float(np.mean((dup_prob >= 0.5) != (base_prob >= 0.5)))
    pd.DataFrame({"actual": yte, "baseline_spam_probability": base_prob,
                  "duplicated_feature_spam_probability": dup_prob,
                  "absolute_probability_shift": np.abs(dup_prob - base_prob)}).to_csv(
                      OUT / "Q5_duplicate_feature_effect.csv", index=False)
    before, after = (base_prob >= 0.5).astype(int), (dup_prob >= 0.5).astype(int)
    pd.DataFrame([
        {"model": "Original features", **metrics(yte, before)},
        {"model": f"Duplicated feature: {duplicated_word}", **metrics(yte, after)}
    ]).to_csv(OUT / "Q5_independence_effect_metrics.csv", index=False)
    plt.figure(figsize=(6, 5))
    plt.scatter(base_prob, dup_prob, alpha=0.45)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Spam probability before duplication")
    plt.ylabel("Spam probability after duplication")
    plt.title(f"Q5: Duplicating '{duplicated_word}'")
    plt.tight_layout()
    plt.savefig(OUT / "Q5_duplicate_feature_effect.png", dpi=150)
    plt.close()
    summary += ["Q5 — Independence assumption",
                "Natural examples of dependence include phrases such as 'call now', 'free entry', and 'claim prize'; phrase meaning and neighboring words are not independent.",
                f"The word '{duplicated_word}' was deliberately duplicated to create perfect dependence.",
                f"Mean absolute probability shift: {shift:.4f}",
                f"Fraction of predictions changed at threshold 0.5: {changed:.2%}",
                "Compare Q5_independence_effect_metrics.csv to see whether the duplicate feature helped or hurt test metrics on this split.",
                "A performance change demonstrates sensitivity to duplicated evidence; natural correlations do not necessarily reduce accuracy in every case.", ""]

    pd.DataFrame(all_rows).to_csv(OUT / "all_questions_metrics.csv", index=False)
    summary += ["OUTPUT FILES",
                "- Q1_scratch_vs_sklearn.csv",
                "- Q2_three_variant_comparison.csv",
                "- Q3_alpha_tuning_summary.csv and alpha CV CSVs",
                "- Q4_word_log_likelihood_ratios.csv, Q4_top_words_by_class.csv, Q4_top_word_log_ratios.png",
                "- Q5_duplicate_feature_effect.csv, Q5_independence_effect_metrics.csv, Q5_duplicate_feature_effect.png",
                "- all_questions_metrics.csv", "- SUMMARY.txt"]
    (OUT / "SUMMARY.txt").write_text("\\n".join(summary) + "\\n", encoding="utf-8")
    print("\\n".join(summary))
    print(f"\\nOutputs saved in: {OUT.resolve()}")

if __name__ == "__main__":
    main()
