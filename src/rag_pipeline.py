from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import INDEX_PATH, TOP_K
from src.text_normalizer import normalize_text


@dataclass
class RetrievedFAQ:
    category: str
    language: str
    question: str
    answer: str
    score: float


class RAGPipeline:
    def __init__(self, index_path=INDEX_PATH):
        payload = joblib.load(index_path)
        self.vectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]
        self.records = payload["records"]

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[RetrievedFAQ]:
        query_vec = self.vectorizer.transform([normalize_text(query)])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedFAQ] = []
        for idx in top_indices:
            row = self.records[int(idx)]
            results.append(
                RetrievedFAQ(
                    category=row["category"],
                    language=row["language"],
                    question=row["question"],
                    answer=row["answer"],
                    score=float(scores[idx]),
                )
            )
        return results