# Product Review Sentiment Analysis (NLP + Machine Learning)

Classifies Flipkart product reviews as **Positive / Neutral / Negative** using TF-IDF features and
a comparison of Logistic Regression, Multinomial Naive Bayes and Linear SVM, with a Streamlit dashboard.

## Run it

```bash
pip install -r requirements.txt
python train.py          # cleans data, trains + compares 3 models, saves artifacts/ (~1.5 min)
python eda.py            # writes the charts to artifacts/charts/
streamlit run app.py     # opens the dashboard
```

The trained model and results are already included in `artifacts/`, so `streamlit run app.py` works immediately.
The original CSV in `data/` is only read, never modified.

## Project layout

| File | Purpose |
|---|---|
| `config.py` | Paths, random seed, column names, label mapping |
| `preprocessing.py` | Text cleaning (shared by training and the app) |
| `data_loader.py` | Loads the CSV, detects columns, cleans, de-duplicates, builds labels |
| `train.py` | Split → TF-IDF → 3 models → cross-validation → evaluation → save with Joblib |
| `eda.py` | Exploratory charts + `eda_summary.json` |
| `model_utils.py` | Loads the model and returns predictions + scores |
| `app.py` | Streamlit app (Home, Predictor, Explorer, Analytics, Model Performance, Batch) |
| `artifacts/` | `sentiment_model.joblib`, `metrics.json`, `reviews_clean.csv.gz`, `charts/` |

## 1. The dataset (inspected, not assumed)

`flipkart_product.csv` — 189,874 rows, 5 columns:

| Column | Meaning | Used as |
|---|---|---|
| `ProductName` | Product title (812 distinct raw names) | product filter / comparison |
| `Price` | Price string, e.g. "₹3,999" (rupee sign corrupted) | converted to a number |
| `Rate` | Star rating, 1–5 | **sentiment label** |
| `Review` | Short headline, e.g. "Wonderful", "Useless product" | **not used** (see below) |
| `Summary` | The review text | **model input** |

Data quality: 5 rows with an invalid rating (text had shifted into the `Rate` column), 14 missing `Summary`,
24,861 fully duplicated rows (25,788 duplicate reviews after validation), and the file is not valid UTF-8 (read as latin-1; emoji
bytes appear as junk characters and are removed during cleaning).

**Labels** (the file has no sentiment column, so they come from the ratings): 1–2★ Negative, 3★ Neutral, 4–5★ Positive.

### Why the `Review` headline is not a feature
The headline almost fully determines the label: grouping by headline, the same sentiment class is
the majority **99.8%** of the time (e.g. "Useless product" is always 1★). A model trained on it would score
very high but simply memorise Flipkart's pre-set headlines, and it would fail on real text typed in the app.
Only `Summary` is used.

## 2. Preprocessing
HTML tags and URLs removed → lower-case → contractions expanded (including "dont", "wont") → repeated letters
shortened ("soooo") → NLTK regex tokenizer (drops digits, punctuation, corrupted emoji bytes) →
stop-word removal that **keeps negations and intensifiers** (`not, no, never, nor, but, very, too, ...`) →
TF-IDF with 1–2-grams so phrases like "not good" and "not bad" are learned.

Duplicates: exact duplicate reviews are removed (163,824 distinct reviews remain for analysis). For modelling,
identical *cleaned texts* are collapsed to one row (79,993 unique texts; 1,165 appeared with different
star ratings and receive the majority label), so the same text can never be in both train and test.

**No data leakage:** the train/test split (80/20, stratified, `random_state=42`) happens first; TF-IDF is fitted inside a
scikit-learn `Pipeline`, i.e. on training data only (and per fold during cross-validation).

## 3. Exploratory findings

| | Count | Share |
|---|---|---|
| Positive | 125,390 | 76.5% |
| Negative | 24,626 | 15.0% |
| Neutral | 13,808 | 8.4% |

- **Strongly imbalanced**: three out of four reviews are positive, and Neutral is rare. Accuracy alone is therefore misleading → **macro-F1** is used for model selection and class weights are used where supported.
- Reviews are very short (median 4 words, mean 8.5). Negative reviews are longest (mean 10.3 words), positive shortest (8.1).
- Frequent words overall: *good, product, very, nice, not, quality, best*. "not", "bad", "worst", "waste" stand out in negative reviews.
- Sentiment differs by product: among products with ≥300 reviews, the share of positive reviews ranges from 40% (a men's sandal) to 96% (MacBook Air M1).
- Charts: `artifacts/charts/` (rating distribution, sentiment distribution, review length, top words, word clouds, sentiment by product, data quality).

## 4. Model results (held-out test set, 15,999 unique reviews)

| Model | Accuracy | Precision (macro) | Recall (macro) | **F1 (macro)** | CV F1 (macro) |
|---|---|---|---|---|---|
| Logistic Regression | 0.832 | 0.689 | 0.726 | 0.700 | 0.684 |
| Multinomial Naive Bayes | 0.860 | 0.711 | 0.648 | 0.666 | 0.661 |
| **Linear SVM (LinearSVC)** | **0.863** | 0.703 | 0.703 | **0.703** | **0.692** |

- **Selected: Linear SVM**, chosen by 3-fold cross-validated macro-F1 on the *training* set (the test set was used once, for reporting).
- Baseline: always predicting "Positive" gives 71.3% accuracy, so 86.3% is a real improvement.
- Per class (Linear SVM): Positive F1 **0.934**, Negative F1 **0.828**, Neutral F1 **0.346**.

## 5. Limitations (please read)
- **Neutral is weak (F1 ≈ 0.35).** Three-star reviews are often written in clearly positive or negative words ("good but..."), so the text alone cannot separate them reliably. This comes from rating-based labels, not only from the model.
- **Labels are star ratings, not human sentiment annotations.** Some reviews have a rating that disagrees with the text.
- Very short reviews ("ok", "good") carry little information.
- **Linear SVM has no true probabilities.** The app shows relative scores (softmax of the decision values), labelled as such.
- Models use fixed, sensible hyper-parameters; no extensive tuning was done.
- Lemmatisation is available (`USE_LEMMATIZATION` in `config.py`, needs `nltk.download('wordnet')`) but is **off** in the shipped results. It was not evaluated, so enable it only together with re-running `python train.py`.
- The model is trained on Flipkart reviews (electronics, home, fashion); it may be less accurate for other domains.
