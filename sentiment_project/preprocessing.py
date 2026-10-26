"""Text preprocessing shared by training, evaluation and the Streamlit app.

The saved model contains `clean_batch` as its first pipeline step, so the app and the
training script are guaranteed to use exactly the same cleaning.
"""
import html
import re
from typing import Iterable, List

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import USE_LEMMATIZATION

# Negations / intensifiers carry sentiment, so they are NOT removed as stop words.
KEEP_WORDS = {
    "not", "no", "nor", "never", "none", "nothing", "nobody", "nowhere", "neither",
    "cannot", "without", "hardly", "but", "very", "too", "less", "more", "most",
    "least", "only", "against",
}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - KEEP_WORDS

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_REPEAT = re.compile(r"(.)\1{2,}")          # "soooo" -> "soo"
_TOKENIZER = RegexpTokenizer(r"[a-z]+")      # NLTK tokenizer, needs no downloaded data

# Contractions, with and without apostrophe (reviews often write "dont", "wont").
_CONTRACTIONS = [
    (r"\bwon'?t\b", "will not"),
    (r"\bcan'?t\b", "can not"),
    (r"\bcannot\b", "can not"),
    (r"\bain'?t\b", "is not"),
    (r"\b(do|does|did|is|are|was|were|has|have|had|would|could|should|must|need)n'?t\b", r"\1 not"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'ve\b", " have"),
    (r"'ll\b", " will"),
    (r"'d\b", " would"),
    (r"'m\b", " am"),
    (r"'s\b", ""),
]
_CONTRACTIONS = [(re.compile(p), r) for p, r in _CONTRACTIONS]

_lemmatizer = None
if USE_LEMMATIZATION:  # pragma: no cover - optional
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()


def clean_text(text) -> str:
    """Clean one review. Returns '' for missing / non-text input."""
    if text is None or not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = _HTML_TAG.sub(" ", t)
    t = _URL.sub(" ", t)
    t = t.lower().replace("’", "'").replace("‘", "'")
    for pattern, repl in _CONTRACTIONS:
        t = pattern.sub(repl, t)
    t = _REPEAT.sub(r"\1\1", t)
    tokens = _TOKENIZER.tokenize(t)          # also drops digits, emoji bytes, punctuation
    tokens = [w for w in tokens if len(w) > 1 and w not in STOP_WORDS]
    if _lemmatizer is not None:
        tokens = [_lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def clean_batch(texts: Iterable) -> List[str]:
    """Vectorised wrapper used as the first step of the sklearn pipeline."""
    return [clean_text(t) for t in texts]
