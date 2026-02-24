# app/services/search_service.py

import numpy as np
from typing import List, Dict
from app.services.knowledge_loader import load_knowledge
from app.services.embedding_service import embed_query


# Load knowledge once at startup
texts, embeddings, metadata = load_knowledge()

# If embeddings not prebuilt → build at runtime
if embeddings is None:
    print("🔧 Building embeddings at runtime...")
    embeddings = np.array([embed_query(text) for text in texts])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_tcode(query: str, top_k: int = 5) -> List[Dict]:
    """Semantic search for SAP TCodes"""

    query_vec = np.array(embed_query(query))

    scores = [
        cosine_similarity(query_vec, emb)
        for emb in embeddings
    ]

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        results.append({
            "tcode": metadata[idx]["tcode"],
            "description": metadata[idx]["description"],
            "score": float(scores[idx])
        })

    return results