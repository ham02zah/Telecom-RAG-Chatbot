import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATA_PATH, INDEX_PATH, REPORTS_DIR, VISUALIZATIONS_DIR
from src.text_normalizer import normalize_text


def create_directories():
    os.makedirs("models", exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


def load_faq_dataset():
    """Load telecom FAQ dataset."""
    df = pd.read_csv(DATA_PATH)

    required_columns = ["category", "language", "question", "answer"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=["question", "answer"])

    df["normalized_question"] = df["question"].apply(normalize_text)
    df["normalized_answer"] = df["answer"].apply(normalize_text)

    df["search_text"] = (
        df["category"].astype(str) + " " +
        df["normalized_question"] + " " +
        df["normalized_answer"]
    )

    return df


def plot_category_distribution(df):
    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x="category", hue="category", palette="Reds_r", legend=False)
    plt.title("FAQ Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/category_distribution.png")
    plt.close()


def plot_language_distribution(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="language", hue="language", palette="Oranges_r", legend=False)
    plt.title("FAQ Language Distribution")
    plt.xlabel("Language")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/language_distribution.png")
    plt.close()


def plot_top_categories(df):
    top_categories = (
        df["category"]
        .value_counts()
        .head(8)
        .reset_index()
    )
    top_categories.columns = ["category", "count"]

    plt.figure(figsize=(9, 5))
    sns.barplot(data=top_categories, x="count", y="category", hue="category", palette="Reds_r", legend=False)
    plt.title("Top FAQ Categories")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/top_categories.png")
    plt.close()


def build_index():
    """Build TF-IDF RAG index and save artifacts."""
    create_directories()

    print("Loading FAQ dataset...")
    df = load_faq_dataset()

    print("Creating visualizations...")
    plot_category_distribution(df)
    plot_language_distribution(df)
    plot_top_categories(df)

    print("Building TF-IDF index...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(df["search_text"])

    chunks = []

    for _, row in df.iterrows():
        chunks.append(
            {
                "category": row["category"],
                "language": row["language"],
                "question": row["question"],
                "answer": row["answer"],
                "search_text": row["search_text"],
            }
        )

    artifacts = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "chunks": chunks,
    }

    joblib.dump(artifacts, INDEX_PATH)

    stats = {
        "total_faqs": int(len(df)),
        "categories": int(df["category"].nunique()),
        "languages": df["language"].unique().tolist(),
        "index_path": INDEX_PATH,
    }

    with open(f"{REPORTS_DIR}/rag_stats.json", "w") as file:
        json.dump(stats, file, indent=4)

    print(f"RAG index saved to {INDEX_PATH}")
    print(f"Stats saved to {REPORTS_DIR}/rag_stats.json")


if __name__ == "__main__":
    build_index()