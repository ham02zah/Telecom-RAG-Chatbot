import joblib
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from src.config import INDEX_PATH, RAG_TOP_K
from src.text_normalizer import normalize_text


def load_index():
    """Load saved RAG artifacts."""
    return joblib.load(INDEX_PATH)


def retrieve_documents(query, top_k=RAG_TOP_K):
    """
    Retrieve most relevant FAQ chunks for a user query.
    """
    artifacts = load_index()

    vectorizer = artifacts["vectorizer"]
    tfidf_matrix = artifacts["tfidf_matrix"]
    chunks = artifacts["chunks"]

    normalized_query = normalize_text(query)

    query_vector = vectorizer.transform([normalized_query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append(
            {
                **chunks[index],
                "score": float(similarity_scores[index]),
            }
        )

    return results