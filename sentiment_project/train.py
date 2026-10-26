"""Train, compare and save the sentiment models.

Run:  python train.py

Leakage control
  * The split into train/test happens BEFORE TF-IDF is fitted; TF-IDF lives inside an
    sklearn Pipeline so it is re-fitted on the training part only (also inside every CV fold).
  * Identical review texts are collapsed to one row before splitting, so the same text can
    not appear in both train and test.
  * The canned 'Review' headline column is not used as a feature (it almost determines the
    star rating) - only the real review text is.
Model selection uses cross-validated macro-F1 on the TRAINING set only; the test set is used
once, for the final report.
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

import config
from data_loader import load_raw, prepare
from preprocessing import clean_batch

LABELS = config.LABELS


def make_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95,
                           sublinear_tf=True, max_features=200_000)


def make_models():
    return {
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced",
                                                  random_state=config.RANDOM_STATE),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM (LinearSVC)": LinearSVC(C=0.5, class_weight="balanced",
                                            random_state=config.RANDOM_STATE),
    }


def evaluate(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro",
                                                 zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="weighted",
                                                  zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "f1_weighted": float(fw),
        "report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def save_confusion_png(cm, title, path):
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(norm, annot=np.array([[f"{c:,}\n({n:.0%})" for c, n in zip(rc, rn)]
                                      for rc, rn in zip(cm, norm)]),
                fmt="", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def top_features(tfidf, clf, k=15):
    if not hasattr(clf, "coef_"):
        return None
    names = np.array(tfidf.get_feature_names_out())
    out = {}
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        out[cls] = [names[j] for j in np.argsort(coefs)[::-1][:k]]
    return out


def main():
    t0 = time.time()
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    config.CHART_DIR.mkdir(exist_ok=True)

    print("Loading and cleaning data ...")
    raw = load_raw()
    analysis, model_df, cols, quality = prepare(raw)
    analysis.to_csv(config.CLEAN_DATA_PATH, index=False, compression="gzip")
    print(f"  raw rows {len(raw):,} | analysis rows {len(analysis):,} | unique texts for modelling {len(model_df):,}")

    X, y = model_df["CleanText"], model_df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    print(f"  train {len(X_train):,} | test {len(X_test):,}")

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    results, fitted = {}, {}
    for name, clf in make_models().items():
        print(f"Training {name} ...")
        pipe = Pipeline([("tfidf", make_tfidf()), ("clf", clf)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
        pipe.fit(X_train, y_train)               # TF-IDF is fitted on the training split only
        pred = pipe.predict(X_test)
        m = evaluate(y_test, pred)
        m["cv_f1_macro_mean"] = float(cv_f1.mean())
        m["cv_f1_macro_std"] = float(cv_f1.std())
        m["supports_probabilities"] = hasattr(clf, "predict_proba")
        results[name], fitted[name] = m, pipe
        save_confusion_png(m["confusion_matrix"], name,
                           config.CHART_DIR / f"confusion_{name.split()[0].lower()}_{name.split()[1].lower()}.png")
        print(f"  CV macro-F1 {m['cv_f1_macro_mean']:.4f} | test acc {m['accuracy']:.4f} | test macro-F1 {m['f1_macro']:.4f}")

    # --- model selection: CV macro-F1 on the training set (not the test set) ---
    best_name = max(results, key=lambda n: results[n]["cv_f1_macro_mean"])
    best_pipe = fitted[best_name]
    print(f"Selected: {best_name}")

    final_clf = best_pipe.named_steps["clf"]
    final_tfidf = best_pipe.named_steps["tfidf"]
    # The saved pipeline is exactly the model that was evaluated (no re-fitting, no calibration),
    # prefixed with the shared cleaning step so it accepts RAW review text.
    score_kind = "probability" if hasattr(final_clf, "predict_proba") else "relative score"

    final_pipeline = Pipeline([
        ("clean", FunctionTransformer(clean_batch, validate=False)),   # same cleaning as training
        ("tfidf", final_tfidf),
        ("clf", final_clf),
    ])
    final_pipeline.named_steps["clean"].fit([""])
    # The saved pipeline takes RAW review text. Verify it on the held-out test texts (already cleaned;
    # cleaning is idempotent for already-clean text) to make sure saved == evaluated model.
    final_pred = final_pipeline.predict(X_test)
    final_metrics = evaluate(y_test, final_pred)

    # Idempotency sanity check of the cleaner
    sample = X_test.iloc[:500].tolist()
    idempotent = float(np.mean([a == b for a, b in zip(clean_batch(sample), sample)]))
    print(f"Cleaner idempotency check on 500 test texts: {idempotent:.1%} unchanged")

    joblib.dump({"pipeline": final_pipeline, "labels": [str(c) for c in final_clf.classes_],
                 "model_name": best_name, "score_kind": score_kind}, config.MODEL_PATH, compress=3)

    # Majority-class baseline for context
    majority = y_train.value_counts().idxmax()
    baseline_acc = float((y_test == majority).mean())

    summary_rows = []
    for n, m in results.items():
        summary_rows.append({"model": n, "accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
                             "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
                             "f1_weighted": m["f1_weighted"], "cv_f1_macro_mean": m["cv_f1_macro_mean"],
                             "cv_f1_macro_std": m["cv_f1_macro_std"]})

    metrics = {
        "labels": LABELS,
        "selected_model": best_name,
        "score_kind": score_kind,
        "selection_criterion": f"mean {config.CV_FOLDS}-fold cross-validated macro-F1 on the training set",
        "comparison": summary_rows,
        "models": results,
        "final_saved_model_test": final_metrics,
        "baseline_majority_accuracy": baseline_acc,
        "split": {"train": int(len(X_train)), "test": int(len(X_test)), "test_size": config.TEST_SIZE,
                  "random_state": config.RANDOM_STATE,
                  "train_class_counts": y_train.value_counts().to_dict(),
                  "test_class_counts": y_test.value_counts().to_dict()},
        "tfidf": {"ngram_range": [1, 2], "min_df": 3, "max_df": 0.95, "sublinear_tf": True,
                  "vocabulary_size": int(len(final_tfidf.vocabulary_))},
        "data_quality": quality,
        "top_features": top_features(final_tfidf, fitted["Logistic Regression"].named_steps["clf"]),
        "label_mapping": {"1-2 stars": "Negative", "3 stars": "Neutral", "4-5 stars": "Positive"},
        "seconds": round(time.time() - t0, 1),
    }
    # Model dataset class distribution (after de-duplication)
    metrics["model_class_counts"] = y.value_counts().to_dict()
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nComparison (held-out test set):")
    print(pd.DataFrame(summary_rows).round(4).to_string(index=False))
    print(f"\nMajority-class baseline accuracy: {baseline_acc:.4f}")
    print(f"Saved model -> {config.MODEL_PATH}\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
