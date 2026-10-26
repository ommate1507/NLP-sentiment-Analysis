"""Exploratory data analysis. Run after train.py:  python eda.py

Reads artifacts/reviews_clean.csv.gz and metrics.json, writes PNG charts to artifacts/charts
and the numeric findings to artifacts/eda_summary.json.
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

PALETTE = {"Negative": "#d1495b", "Neutral": "#edae49", "Positive": "#2e8b57"}
ORDER = ["Negative", "Neutral", "Positive"]
sns.set_theme(style="whitegrid", context="notebook")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(config.CHART_DIR / name, dpi=150)
    plt.close(fig)


def top_words(series, n=20):
    c = Counter(w for t in series for w in t.split())
    return c.most_common(n)


def main():
    config.CHART_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.CLEAN_DATA_PATH, keep_default_na=False, na_values=[""])
    df["CleanText"] = df["CleanText"].fillna("")
    metrics = json.load(open(config.METRICS_PATH))
    dq = metrics["data_quality"]
    summary = {}

    # 1. Data quality (raw file)
    raw = dq["raw"]
    items = {**{f"missing: {k}": v for k, v in raw["missing"].items() if v},
             "invalid rating rows": dq["rows_dropped_invalid_rating"],
             "exact duplicate reviews": dq["rows_dropped_exact_duplicates"]}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    s = pd.Series(items).sort_values()
    ax.barh(s.index, s.values, color="#4c78a8")
    ax.set_xscale("log"); ax.set_xlabel("rows (log scale)"); ax.set_title("Data quality issues in the raw file")
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:,}", va="center")
    save(fig, "01_data_quality.png")

    # 2. Rating distribution
    r = df["Rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(x=r.index, y=r.values, color="#4c78a8", ax=ax)
    for i, v in enumerate(r.values):
        ax.text(i, v, f"{v:,}\n{v / r.sum():.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Star rating"); ax.set_ylabel("Reviews"); ax.set_title("Rating distribution")
    ax.set_ylim(0, r.max() * 1.18)
    save(fig, "02_rating_distribution.png")

    # 3. Sentiment distribution (analysis dataset, and the de-duplicated modelling dataset)
    s_all = df["Sentiment"].value_counts().reindex(ORDER)
    s_model = pd.Series(metrics["model_class_counts"]).reindex(ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, s_, ttl in zip(axes, (s_all, s_model), ("All distinct reviews", "Unique review texts (used for modelling)")):
        ax.bar(ORDER, s_.values, color=[PALETTE[k] for k in ORDER])
        for i, v in enumerate(s_.values):
            ax.text(i, v, f"{v:,}\n{v / s_.sum():.1%}", ha="center", va="bottom", fontsize=9)
        ax.set_title(ttl); ax.set_ylim(0, s_.max() * 1.2)
    axes[0].set_ylabel("Reviews")
    fig.suptitle("Sentiment distribution (from star ratings)")
    save(fig, "03_sentiment_distribution.png")
    summary["sentiment_counts"] = {k: int(v) for k, v in s_all.items()}
    summary["sentiment_pct"] = {k: round(float(v) / s_all.sum() * 100, 1) for k, v in s_all.items()}
    summary["total_reviews"] = int(len(df))
    summary["n_products"] = int(df["ProductShort"].nunique())

    # 4. Review length
    d = df[df["WordCount"] <= df["WordCount"].quantile(0.99)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(d, x="WordCount", bins=40, color="#4c78a8", ax=axes[0])
    axes[0].set_title("Review length (words, 99th percentile cut)")
    sns.boxplot(data=d, x="Sentiment", y="WordCount", order=ORDER, palette=PALETTE, hue="Sentiment",
                legend=False, showfliers=False, ax=axes[1])
    axes[1].set_title("Review length by sentiment")
    save(fig, "04_review_length.png")
    summary["words_median"] = float(df["WordCount"].median())
    summary["words_mean"] = round(float(df["WordCount"].mean()), 1)
    summary["words_median_by_sentiment"] = df.groupby("Sentiment")["WordCount"].median().to_dict()
    summary["words_mean_by_sentiment"] = df.groupby("Sentiment")["WordCount"].mean().round(1).to_dict()

    # 5. Most frequent words (overall and per sentiment)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    summary["top_words"] = {}
    for ax, sent in zip(axes, ORDER):
        tw = top_words(df.loc[df["Sentiment"] == sent, "CleanText"], 15)
        summary["top_words"][sent] = tw
        words, counts = zip(*tw)
        ax.barh(words[::-1], counts[::-1], color=PALETTE[sent])
        ax.set_title(f"{sent} reviews")
    fig.suptitle("Most frequent words (after cleaning)")
    save(fig, "05_top_words_by_sentiment.png")
    summary["top_words"]["Overall"] = top_words(df["CleanText"], 20)

    # 6. Word clouds
    try:
        from wordcloud import WordCloud
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, sent in zip(axes, ORDER):
            text = " ".join(df.loc[df["Sentiment"] == sent, "CleanText"].sample(
                min(30000, (df["Sentiment"] == sent).sum()), random_state=config.RANDOM_STATE))
            wc = WordCloud(width=700, height=450, background_color="white", colormap="viridis",
                           max_words=80, random_state=config.RANDOM_STATE).generate(text)
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(sent)
        save(fig, "06_wordclouds.png")
    except ImportError:
        print("wordcloud not installed - skipping word clouds (pip install wordcloud)")

    # 7. Sentiment by product (products with >= 300 reviews)
    cnt = df.groupby("ProductShort").size()
    big = cnt[cnt >= 300].index
    pt = (df[df["ProductShort"].isin(big)].groupby(["ProductShort", "Sentiment"]).size()
          .unstack(fill_value=0).reindex(columns=ORDER, fill_value=0))
    pt_pct = pt.div(pt.sum(axis=1), axis=0) * 100
    pt_pct["n"] = pt.sum(axis=1)
    top = pt_pct.sort_values("n", ascending=False).head(15).sort_values("Positive")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    left = np.zeros(len(top))
    for sent in ORDER:
        ax.barh(top.index.str.slice(0, 45), top[sent], left=left, color=PALETTE[sent], label=sent)
        left += top[sent].values
    ax.set_xlabel("% of reviews")
    ax.set_title("Sentiment mix of the 15 most-reviewed products", pad=34)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    save(fig, "07_sentiment_by_product.png")
    summary["n_products_ge_300_reviews"] = int(len(big))
    ranked = pt_pct[pt_pct["n"] >= 300].sort_values("Positive")
    summary["lowest_positive_share_products"] = ranked.head(3)["Positive"].round(1).to_dict()
    summary["highest_positive_share_products"] = ranked.tail(3)["Positive"].round(1).to_dict()

    # 8. Price vs sentiment
    p = df.dropna(subset=["Price"])
    summary["median_price_by_sentiment"] = p.groupby("Sentiment")["Price"].median().to_dict()

    with open(config.ARTIFACT_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("Charts written to", config.CHART_DIR)
    print(json.dumps({k: v for k, v in summary.items() if k != "top_words"}, indent=1, default=float))
    print("top words:", {k: [w for w, _ in v[:8]] for k, v in summary["top_words"].items()})


if __name__ == "__main__":
    main()
