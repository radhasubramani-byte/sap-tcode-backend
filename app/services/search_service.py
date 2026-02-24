from typing import List
from openai import OpenAI
import numpy as np
from app.services.knowledge_loader import load_knowledge

client = OpenAI()

# load embeddings + metadata at startup
index, metadata = load_knowledge()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embed_query(query: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding


def search_tcode(user_query: str, top_k: int = 5) -> List[dict]:
    """
    Semantic search SAP TCodes
    """
    query_vector = embed_query(user_query)

    similarities = []
    for i, item in enumerate(metadata):
        sim = cosine_similarity(query_vector, item["embedding"])
        similarities.append((sim, item))

    similarities.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, item in similarities[:top_k]:
        results.append({
            "tcode": item["tcode"],
            "description": item["description"],
            "score": float(score)
        })

    return results