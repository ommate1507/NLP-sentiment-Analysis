"""Helpers for using the saved model (shared by the Streamlit app and any script)."""
from typing import List, Tuple

import joblib
import numpy as np

import config


def load_model():
    """Returns dict with keys: pipeline (raw text -> label), labels, model_name, score_kind."""
    return joblib.load(config.MODEL_PATH)


def predict_with_scores(bundle, texts: List[str]) -> Tuple[np.ndarray, np.ndarray, str]:
    """Predict labels and per-class scores for RAW review texts.

    Returns (labels, scores, score_kind). score_kind is 'probability' for models with
    predict_proba (Logistic Regression, Naive Bayes) and 'relative score' for LinearSVC, where
    scores are a softmax over the SVM decision values (they sum to 1 but are not calibrated).
    """
    pipe = bundle["pipeline"]
    labels = pipe.predict(texts)
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        scores, kind = pipe.predict_proba(texts), "probability"
    else:
        d = pipe.decision_function(texts)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        scores, kind = e / e.sum(axis=1, keepdims=True), "relative score"
    return labels, scores, kind
