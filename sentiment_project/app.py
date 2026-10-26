"""Streamlit dashboard for the Flipkart review sentiment project.

Run:  streamlit run app.py      (after python train.py and python eda.py)
"""
import io
import json
from collections import Counter

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import config
from model_utils import load_model, predict_with_scores

ORDER = config.LABELS
COLORS = {"Negative": "#d1495b", "Neutral": "#edae49", "Positive": "#2e8b57"}
COLOR_SCALE = alt.Scale(domain=ORDER, range=[COLORS[k] for k in ORDER])
EMOJI = {"Negative": "😞", "Neutral": "😐", "Positive": "😊"}

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #1f3a5f 0%, #3b6ea5 100%); color: white;
       padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0 0 .3rem 0; font-size: 2rem;}
.hero p {margin: 0; opacity: .92; font-size: 1.02rem;}
.result-card {border-radius: 14px; padding: 1.2rem 1.5rem; color: white; margin: .5rem 0 1rem 0;}
.result-card h2 {color: white; margin: 0; font-size: 2rem;}
.result-card span {opacity: .9;}
div[data-testid="stMetric"] {background: rgba(120,140,170,.10); padding: .8rem 1rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading model ...")
def get_model():
    return load_model()


@st.cache_data(show_spinner="Loading dataset ...")
def get_data() -> pd.DataFrame:
    df = pd.read_csv(config.CLEAN_DATA_PATH, keep_default_na=False, na_values=[""])
    df["CleanText"] = df["CleanText"].fillna("")
    df["Sentiment"] = pd.Categorical(df["Sentiment"], categories=ORDER, ordered=True)
    return df


@st.cache_data
def get_metrics() -> dict:
    with open(config.METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def get_eda() -> dict:
    p = config.ARTIFACT_DIR / "eda_summary.json"
    return json.load(open(p)) if p.exists() else {}


@st.cache_data(show_spinner="Counting words ...")
def word_counts(sentiment: str, n: int):
    df = get_data()
    s = df["CleanText"] if sentiment == "All" else df.loc[df["Sentiment"] == sentiment, "CleanText"]
    return Counter(w for t in s for w in t.split()).most_common(n)


def need_artifacts():
    missing = [p.name for p in (config.MODEL_PATH, config.METRICS_PATH, config.CLEAN_DATA_PATH) if not p.exists()]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}. Run `python train.py` (then `python eda.py`) first.")
        st.stop()


def bar_sentiment(counts: pd.Series, title: str = ""):
    d = counts.reindex(ORDER).fillna(0).rename_axis("Sentiment").reset_index(name="Reviews")
    d["Share"] = d["Reviews"] / d["Reviews"].sum()
    chart = alt.Chart(d, title=title).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("Sentiment:N", sort=ORDER, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Reviews:Q"),
        color=alt.Color("Sentiment:N", scale=COLOR_SCALE, legend=None),
        tooltip=["Sentiment", alt.Tooltip("Reviews:Q", format=","), alt.Tooltip("Share:Q", format=".1%")])
    return chart.properties(height=300)


def header(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- pages
def page_home():
    df, m, eda = get_data(), get_metrics(), get_eda()
    header("💬 Product Review Sentiment Analyzer",
           "An NLP + machine-learning system that classifies customer reviews as Positive, Neutral or Negative "
           "(trained on Flipkart product reviews, labels derived from star ratings).")
    sel = m["selected_model"]
    final = m["final_saved_model_test"]
    c = st.columns(5)
    c[0].metric("Distinct reviews", f"{len(df):,}")
    c[1].metric("Products", f"{df['ProductShort'].nunique():,}")
    c[2].metric("Average rating", f"{df['Rating'].mean():.2f} ★")
    c[3].metric("Test accuracy", f"{final['accuracy']:.1%}")
    c[4].metric("Test macro-F1", f"{final['f1_macro']:.3f}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Sentiment distribution")
        st.altair_chart(bar_sentiment(df["Sentiment"].value_counts()), width="stretch")
    with right:
        st.subheader("Key facts")
        share = df["Sentiment"].value_counts(normalize=True)
        st.markdown(f"""
- **Positive** {share['Positive']:.1%} · **Neutral** {share['Neutral']:.1%} · **Negative** {share['Negative']:.1%}
- Median review length: **{int(df['WordCount'].median())} words**
- Selected model: **{sel}** (chosen by cross-validated macro-F1)
- Train / test texts: **{m['split']['train']:,} / {m['split']['test']:,}**
- Always predicting “Positive” would score **{m['baseline_majority_accuracy']:.1%}** accuracy, so macro-F1 is the fairer measure
""")
        st.info("The classes are imbalanced and **Neutral** (3-star) reviews are the hardest to detect — "
                "see the Model Performance page.")

    st.subheader("How it works")
    st.markdown("`Raw review` → **cleaning** (HTML/URLs, contractions, negation-safe stop words) → "
                "**TF-IDF** (1–2-grams) → **Linear SVM / Logistic Regression / Naive Bayes** → sentiment")


def page_predict():
    need_artifacts()
    bundle = get_model()
    header("🔍 Review Sentiment Predictor", "Type or paste a customer review and the model will classify it.")
    examples = {
        "— choose an example —": "",
        "Positive": "Excellent product, works perfectly and totally worth the money!",
        "Negative": "Worst purchase ever. It stopped working in two days, don't buy this.",
        "Neutral": "It's okay, average quality for the price. Nothing special.",
        "Negation": "The sound quality is not good at all.",
    }
    ex = st.selectbox("Try an example", list(examples), index=0)
    if ex != "— choose an example —":
        st.session_state["review_text"] = examples[ex]
    text = st.text_area("Customer review", key="review_text", height=150,
                        placeholder="e.g. The battery lasts all day and the screen is great ...")
    if st.button("Predict Sentiment", type="primary"):
        if not text or not text.strip():
            st.error("Please enter a review first — the text box is empty.")
            return
        if not any(ch.isalpha() for ch in text):
            st.error("The review contains no letters. Please enter some text.")
            return
        from preprocessing import clean_text
        if not clean_text(text):
            st.warning("After cleaning, nothing is left to analyse (only stop words, numbers or emoji). "
                       "Try a longer review.")
            return
        labels, scores, kind = predict_with_scores(bundle, [text])
        label, sc = labels[0], scores[0]
        classes = list(bundle["labels"])
        st.markdown(f'<div class="result-card" style="background:{COLORS[label]}">'
                    f'<span>Predicted sentiment</span><h2>{EMOJI[label]} {label}</h2></div>',
                    unsafe_allow_html=True)
        d = pd.DataFrame({"Sentiment": classes, "Score": sc})
        ch = alt.Chart(d).mark_bar().encode(
            y=alt.Y("Sentiment:N", sort=ORDER, title=None),
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=kind.capitalize()),
            color=alt.Color("Sentiment:N", scale=COLOR_SCALE, legend=None),
            tooltip=["Sentiment", alt.Tooltip("Score:Q", format=".1%")]).properties(height=140)
        st.altair_chart(ch, width="stretch")
        if kind == "probability":
            st.caption(f"Class probabilities from **{bundle['model_name']}**.")
        else:
            st.caption(f"**{bundle['model_name']}** does not output true probabilities. The bars show relative "
                       "scores (a softmax of the SVM decision values) — useful for comparing classes, "
                       "but not a calibrated probability.")
        if len(text.split()) < 3:
            st.caption("⚠️ Very short reviews give less reliable predictions.")


def page_explorer():
    need_artifacts()
    df = get_data()
    header("🗂️ Dataset Explorer", "Browse, search and filter the cleaned review dataset.")
    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        ratings = c1.multiselect("Star rating", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        sents = c2.multiselect("Sentiment", ORDER, default=ORDER)
        prod_q = c3.text_input("Product name contains")
        c4, c5 = st.columns([2, 1])
        kw = c4.text_input("Review text contains")
        pmin, pmax = float(df["Price"].min()), float(df["Price"].max())
        price = c5.slider("Price (₹)", pmin, pmax, (pmin, pmax))
    f = df[df["Rating"].isin(ratings) & df["Sentiment"].isin(sents)]
    f = f[(f["Price"].between(*price)) | f["Price"].isna()]
    if prod_q:
        f = f[f["ProductName"].str.contains(prod_q, case=False, na=False, regex=False)]
    if kw:
        f = f[f["ReviewText"].str.contains(kw, case=False, na=False, regex=False)]
    st.markdown(f"**{len(f):,}** of {len(df):,} reviews match.")
    if f.empty:
        st.warning("No reviews match these filters.")
        return
    show = f[["ProductShort", "Rating", "Sentiment", "ReviewText", "Price", "WordCount"]].rename(
        columns={"ProductShort": "Product", "ReviewText": "Review", "WordCount": "Words"})
    n = st.slider("Rows to display", 10, min(2000, len(show)), min(200, len(show)))
    st.dataframe(show.head(n), width="stretch", hide_index=True, height=420)
    st.download_button("Download filtered rows (CSV)", f.drop(columns=["CleanText"]).to_csv(index=False).encode("utf-8"),
                       "filtered_reviews.csv", "text/csv")
    st.subheader("Summary statistics")
    a, b = st.columns(2)
    a.dataframe(f[["Rating", "Price", "WordCount", "CharCount"]].describe().round(2), width="stretch")
    b.altair_chart(bar_sentiment(f["Sentiment"].value_counts(), "Sentiment in current selection"),
                   width="stretch")


def page_analytics():
    need_artifacts()
    df = get_data()
    header("📊 Analytics", "Ratings, sentiment, vocabulary and product-level comparisons.")
    t1, t2, t3, t4 = st.tabs(["Sentiment & ratings", "Common words", "Products", "Review length"])
    with t1:
        c1, c2 = st.columns(2)
        c1.altair_chart(bar_sentiment(df["Sentiment"].value_counts(), "Sentiment counts"), width="stretch")
        r = df["Rating"].value_counts().sort_index().rename_axis("Rating").reset_index(name="Reviews")
        c2.altair_chart(alt.Chart(r, title="Rating distribution").mark_bar(color="#4c78a8").encode(
            x=alt.X("Rating:O", axis=alt.Axis(labelAngle=0, title="Stars")), y="Reviews:Q",
            tooltip=["Rating", alt.Tooltip("Reviews:Q", format=",")]).properties(height=300),
            width="stretch")
        pct = (df["Sentiment"].value_counts(normalize=True).reindex(ORDER) * 100).round(1)
        st.dataframe(pd.DataFrame({"Share (%)": pct, "Reviews": df["Sentiment"].value_counts().reindex(ORDER)}),
                     width="stretch")
    with t2:
        c1, c2 = st.columns([1, 1])
        sent = c1.selectbox("Sentiment", ["All"] + ORDER)
        n = c2.slider("Number of words", 10, 40, 20)
        wc = pd.DataFrame(word_counts(sent, n), columns=["Word", "Count"])
        color = "#4c78a8" if sent == "All" else COLORS[sent]
        st.altair_chart(alt.Chart(wc).mark_bar(color=color).encode(
            y=alt.Y("Word:N", sort="-x", title=None), x="Count:Q",
            tooltip=["Word", alt.Tooltip("Count:Q", format=",")]).properties(height=22 * n + 30),
            width="stretch")
        wcloud = config.CHART_DIR / "06_wordclouds.png"
        if wcloud.exists():
            st.image(str(wcloud), caption="Word clouds by sentiment (generated by eda.py)")
    with t3:
        c1, c2, c3 = st.columns(3)
        min_n = c1.slider("Minimum reviews per product", 50, 1000, 300, step=50)
        top_n = c2.slider("Products to show", 5, 30, 15)
        sort_by = c3.selectbox("Sort by", ["Most reviews", "Highest % positive", "Highest % negative"])
        g = (df.groupby(["ProductShort", "Sentiment"], observed=True).size().unstack(fill_value=0)
             .reindex(columns=ORDER, fill_value=0))
        g["Total"] = g.sum(axis=1)
        g = g[g["Total"] >= min_n]
        if g.empty:
            st.warning("No product has that many reviews — lower the minimum.")
        else:
            pct = g[ORDER].div(g["Total"], axis=0) * 100
            order = {"Most reviews": g["Total"], "Highest % positive": pct["Positive"],
                     "Highest % negative": pct["Negative"]}[sort_by].sort_values(ascending=False).head(top_n).index
            long = pct.loc[order].reset_index().melt("ProductShort", var_name="Sentiment", value_name="Percent")
            long["Product"] = long["ProductShort"].str.slice(0, 45)
            st.altair_chart(alt.Chart(long).mark_bar().encode(
                y=alt.Y("Product:N", sort=[p[:45] for p in order], title=None),
                x=alt.X("Percent:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of reviews"),
                color=alt.Color("Sentiment:N", scale=COLOR_SCALE, legend=alt.Legend(orient="top")),
                order=alt.Order("Sentiment:N", sort="ascending"),
                tooltip=["Product", "Sentiment", alt.Tooltip("Percent:Q", format=".1f")]).properties(
                height=24 * len(order) + 40), width="stretch")
            st.caption(f"{len(g)} products have at least {min_n} reviews.")
    with t4:
        cap = int(df["WordCount"].quantile(0.99))
        d = df[df["WordCount"] <= cap]
        st.altair_chart(alt.Chart(d.sample(min(len(d), 60000), random_state=1)).mark_bar().encode(
            x=alt.X("WordCount:Q", bin=alt.Bin(maxbins=40), title="Words per review"),
            y=alt.Y("count()", title="Reviews"), color=alt.Color("Sentiment:N", scale=COLOR_SCALE),
            tooltip=["count()"]).properties(height=320), width="stretch")
        st.dataframe(df.groupby("Sentiment", observed=True)["WordCount"].describe().round(1), width="stretch")
        st.caption(f"Showing reviews up to {cap} words (99th percentile). Negative reviews tend to be longer.")


def page_performance():
    need_artifacts()
    m = get_metrics()
    header("🏆 Model Performance", "All numbers are computed on the held-out test set of your dataset.")
    comp = pd.DataFrame(m["comparison"]).set_index("model")
    show = comp.rename(columns={"accuracy": "Accuracy", "precision_macro": "Precision (macro)",
                                "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)",
                                "f1_weighted": "F1 (weighted)", "cv_f1_macro_mean": "CV F1 (macro)",
                                "cv_f1_macro_std": "CV std"})
    st.dataframe(show.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8d5"), width="stretch")
    sel = m["selected_model"]
    st.success(f"**Selected model: {sel}.** It has the highest {m['selection_criterion']}. "
               "Macro-F1 weights the three classes equally, so a model cannot win simply by predicting "
               "“Positive” for everything.")
    st.caption(f"Reference: always predicting the majority class gives {m['baseline_majority_accuracy']:.1%} accuracy. "
               f"Train/test split: {m['split']['train']:,}/{m['split']['test']:,} unique texts, stratified, "
               f"random_state={m['split']['random_state']}.")

    st.subheader("Confusion matrix & per-class report")
    name = st.selectbox("Model", list(m["models"]), index=list(m["models"]).index(sel))
    mm = m["models"][name]
    c1, c2 = st.columns([1, 1])
    cm = np.array(mm["confusion_matrix"])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(norm, annot=np.array([[f"{c:,}\n({n:.0%})" for c, n in zip(rc, rn)] for rc, rn in zip(cm, norm)]),
                fmt="", cmap="Blues", xticklabels=ORDER, yticklabels=ORDER, cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
    c1.pyplot(fig, width="stretch"); plt.close(fig)
    rep = pd.DataFrame(mm["report"]).T.loc[ORDER + ["macro avg", "weighted avg"]]
    rep["support"] = rep["support"].astype(int)
    c2.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}),
                 width="stretch")
    c2.markdown("**Reading this:** Positive and Negative reviews are detected well. **Neutral** (3-star) reviews "
                "are often written in positive or negative words, so even a good model confuses them — "
                "this is a limit of rating-based labels, not just of the model.")

    if m.get("top_features"):
        st.subheader("Most indicative words and phrases (Logistic Regression)")
        cols = st.columns(3)
        for col, cls in zip(cols, ORDER):
            col.markdown(f"**{cls}**")
            col.write(", ".join(m["top_features"][cls][:12]))

    with st.expander("Data preparation summary"):
        q = m["data_quality"]
        st.markdown(f"""
- Raw rows: **{q['raw']['rows']:,}** · exact duplicate rows in the raw file: **{q['raw']['duplicate_rows']:,}**
- Dropped: {q['rows_dropped_invalid_rating']} rows with an invalid rating, {q['rows_dropped_missing_text']} with no review text,
  {q['rows_dropped_exact_duplicates']:,} duplicate reviews, {q['rows_dropped_empty_after_cleaning']} empty after cleaning
- Distinct reviews kept for analysis: **{q['analysis_rows']:,}**
- Unique cleaned texts used for modelling: **{q['model_rows']:,}** ({q['texts_with_conflicting_labels']:,} texts that appeared with different star ratings were given their majority label)
- Labels: 1–2★ Negative · 3★ Neutral · 4–5★ Positive
""")


def page_batch():
    need_artifacts()
    bundle = get_model()
    header("📁 Batch Prediction", "Upload a CSV of reviews and download the predicted sentiments.")
    up = st.file_uploader("CSV file", type=["csv"])
    st.download_button("Download a sample CSV", "review\nGreat product, loved it\nVery bad quality, not worth it\nIt is okay\n",
                       "sample_reviews.csv", "text/csv")
    if up is None:
        st.info("Upload a CSV with one review per row. You will be asked which column holds the review text.")
        return
    raw = up.getvalue()
    df = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            st.error(f"Could not read this CSV: {e}")
            return
    if df is None or df.empty or df.shape[1] == 0:
        st.error("The file is empty or could not be read.")
        return
    st.caption(f"{len(df):,} rows, {df.shape[1]} columns")
    guess = next((c for c in df.columns if str(c).lower() in ("review", "reviews", "summary", "text",
                                                                "review_text", "reviewtext", "comment", "content")),
                 None)
    if guess is None:
        obj = [c for c in df.columns if df[c].dtype == object]
        guess = max(obj, key=lambda c: df[c].astype(str).str.len().mean()) if obj else df.columns[0]
    col = st.selectbox("Column containing the review text", list(df.columns), index=list(df.columns).index(guess))
    st.dataframe(df.head(5), width="stretch", hide_index=True)
    if st.button("Predict sentiment for all rows", type="primary"):
        texts = df[col].fillna("").astype(str)
        valid = texts.str.strip().str.len() > 0
        if not valid.any():
            st.error(f"Column '{col}' has no text to analyse.")
            return
        with st.spinner("Predicting ..."):
            labels, scores, kind = predict_with_scores(bundle, texts[valid].tolist())
        out = df.copy()
        out["predicted_sentiment"] = None
        out.loc[valid, "predicted_sentiment"] = labels
        for i, cls in enumerate(bundle["labels"]):
            out[f"score_{cls.lower()}"] = np.nan
            out.loc[valid, f"score_{cls.lower()}"] = scores[:, i].round(4)
        if (~valid).any():
            st.warning(f"{int((~valid).sum())} row(s) had empty text and were left without a prediction.")
        st.success(f"Predicted {int(valid.sum()):,} reviews.")
        a, b = st.columns([2, 1])
        a.dataframe(out, width="stretch", hide_index=True, height=380)
        b.altair_chart(bar_sentiment(out["predicted_sentiment"].value_counts(), "Predicted sentiment"),
                       width="stretch")
        st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                           "sentiment_predictions.csv", "text/csv", type="primary")
        if kind != "probability":
            st.caption("score_* columns are relative scores (softmax of SVM decision values), not calibrated probabilities.")


# ----------------------------------------------------------------------------- router
PAGES = {
    "🏠 Home": page_home,
    "🔍 Sentiment Predictor": page_predict,
    "🗂️ Dataset Explorer": page_explorer,
    "📊 Analytics": page_analytics,
    "🏆 Model Performance": page_performance,
    "📁 Batch Prediction": page_batch,
}
with st.sidebar:
    st.title("Navigation")
    choice = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("NLP · TF-IDF · scikit-learn · Streamlit")

need_artifacts()
PAGES[choice]()
