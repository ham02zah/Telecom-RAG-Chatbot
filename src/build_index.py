import csv

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATA_PATH, INDEX_PATH
from src.text_normalizer import normalize_text


def load_faq_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="error",
    )
    required = {"category", "language", "question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["question", "answer"]).copy()
    df["search_text"] = (
        df["category"].astype(str)
        + " "
        + df["question"].astype(str)
        + " "
        + df["answer"].astype(str)
    ).map(normalize_text)
    return df.reset_index(drop=True)


def build_index() -> None:
    print("Loading FAQ dataset...")
    df = load_faq_dataset()
    print(f"Loaded {len(df)} FAQ rows")

    print("Building TF-IDF index...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(df["search_text"])

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "records": df.to_dict(orient="records"),
    }
    joblib.dump(payload, INDEX_PATH)
    print(f"Saved index to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()