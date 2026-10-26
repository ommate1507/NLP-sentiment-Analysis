"""Central configuration for the Flipkart review sentiment project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "flipkart_product.csv"   # original file, never modified
ARTIFACT_DIR = BASE_DIR / "artifacts"
CHART_DIR = ARTIFACT_DIR / "charts"
MODEL_PATH = ARTIFACT_DIR / "sentiment_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
CLEAN_DATA_PATH = ARTIFACT_DIR / "reviews_clean.csv.gz"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3

# The CSV contains stray bytes (corrupted rupee sign / emoji), so it is not valid UTF-8.
CSV_ENCODING = "latin-1"

# Column names found by inspecting the dataset. If a name is not present in the file,
# data_loader.detect_columns() falls back to heuristics, so the code adapts to similar data.
COLUMN_OVERRIDES = {
    "text": "Summary",        # the review body (what the customer actually wrote)
    "rating": "Rate",         # 1-5 stars
    "product": "ProductName",
    "price": "Price",
    "title": "Review",        # short canned headline, e.g. "Wonderful" -> NOT used for training (see README)
}

# Sentiment labels derived from star ratings (1-5 scale confirmed in the data)
LABELS = ["Negative", "Neutral", "Positive"]


def rating_to_sentiment(rating: float) -> str:
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


# Optional WordNet lemmatisation. Needs the NLTK 'wordnet' corpus (nltk.download('wordnet')).
# The shipped results were produced with this OFF; if you turn it on, re-run train.py.
USE_LEMMATIZATION = False
