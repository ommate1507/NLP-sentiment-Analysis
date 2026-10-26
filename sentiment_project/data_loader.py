"""Loading, inspecting and preparing the review dataset (the original CSV is never modified)."""
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config
from preprocessing import clean_batch


def load_raw(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.DATA_PATH, encoding=config.CSV_ENCODING, dtype=str)


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map logical roles -> actual column names. Uses config overrides when the column exists,
    otherwise falls back to simple heuristics so the project adapts to a similar dataset."""
    cols = {}
    for role, name in config.COLUMN_OVERRIDES.items():
        if name in df.columns:
            cols[role] = name

    if "rating" not in cols:
        for c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) > 0.9 * len(df) and vals.between(0, 5).all() and vals.nunique() <= 10:
                cols["rating"] = c
                break
    if "text" not in cols:
        obj = [c for c in df.columns if c != cols.get("rating") and df[c].dtype == object]
        if obj:
            cols["text"] = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    if "product" not in cols:
        for c in df.columns:
            if re.search(r"product|item|name|asin", c, re.I) and c != cols.get("text"):
                cols["product"] = c
                break
    missing = [r for r in ("text", "rating") if r not in cols]
    if missing:
        raise ValueError(f"Could not identify required column(s): {missing}. Columns: {list(df.columns)}")
    return cols


def profile(df: pd.DataFrame) -> dict:
    """Raw-file profile (before any cleaning)."""
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _clean_product(name) -> str:
    if not isinstance(name, str):
        return ""
    s = re.sub(r"[^\x20-\x7e]+", " ", name)       # drop corrupted bytes
    s = re.sub(r"\s*\?+\s*(?=\()", " ", s)          # leftover '??' before the '(' variant block
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s*\?+$", "", s)                # trailing '?' left by a corrupted byte


def _price_to_number(p) -> float:
    if not isinstance(p, str):
        return np.nan
    digits = re.sub(r"[^\d.]", "", p.replace(",", ""))
    try:
        return float(digits) if digits else np.nan
    except ValueError:
        return np.nan


def prepare(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Return (analysis_df, model_df, columns, quality_report).

    analysis_df : one row per distinct review (exact duplicates removed) with all metadata
                  -> used for EDA, the dashboard and the dataset explorer.
    model_df    : one row per distinct *cleaned review text* with a majority-vote label
                  -> used for training/evaluation, so identical texts can never sit in both
                     the train and the test split.
    """
    cols = detect_columns(df_raw)
    q = {"raw": profile(df_raw), "columns_used": cols}

    df = df_raw.copy()
    df["_rating"] = pd.to_numeric(df[cols["rating"]], errors="coerce")
    bad_rating = df["_rating"].isna() | ~df["_rating"].between(1, 5)
    q["rows_dropped_invalid_rating"] = int(bad_rating.sum())
    df = df[~bad_rating]

    no_text = df[cols["text"]].isna() | (df[cols["text"]].astype(str).str.strip() == "")
    q["rows_dropped_missing_text"] = int(no_text.sum())
    df = df[~no_text]

    dedupe_keys = [c for c in (cols.get("product"), cols.get("title"), cols["text"], cols["rating"]) if c]
    n_before = len(df)
    df = df.drop_duplicates(subset=dedupe_keys).reset_index(drop=True)
    q["rows_dropped_exact_duplicates"] = int(n_before - len(df))

    out = pd.DataFrame({
        "ProductName": df[cols["product"]].map(_clean_product) if "product" in cols else "",
        "Price": df[cols["price"]].map(_price_to_number) if "price" in cols else np.nan,
        "Rating": df["_rating"].astype(int),
        "ReviewText": df[cols["text"]].astype(str).str.strip(),
    })
    out["ProductShort"] = (out["ProductName"].str.split("(").str[0].str.strip()
                           .str.replace(r"\s*\?+$", "", regex=True).str.slice(0, 60))
    out["Sentiment"] = out["Rating"].map(config.rating_to_sentiment)
    out["CleanText"] = clean_batch(out["ReviewText"])
    out["WordCount"] = out["ReviewText"].str.split().str.len()
    out["CharCount"] = out["ReviewText"].str.len()

    empty_clean = out["CleanText"].str.len() == 0
    q["rows_dropped_empty_after_cleaning"] = int(empty_clean.sum())
    analysis = out[~empty_clean].reset_index(drop=True)

    # Model dataset: unique cleaned text, majority label (ties -> the lower star label)
    grp = analysis.groupby("CleanText")["Sentiment"]
    n_unique = grp.ngroups
    majority = grp.agg(lambda s: s.value_counts().sort_index().idxmax())
    conflicts = int((grp.nunique() > 1).sum())
    model_df = majority.rename("Sentiment").reset_index()
    q["unique_clean_texts"] = int(n_unique)
    q["texts_with_conflicting_labels"] = conflicts
    q["analysis_rows"] = int(len(analysis))
    q["model_rows"] = int(len(model_df))
    return analysis, model_df, cols, q
