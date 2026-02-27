import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss

from app.services.knowledge_loader import load_knowledge


# -------------------------------
# Configuration (tunable logic)
# -------------------------------
HIGH_CONFIDENCE = 0.72
MEDIUM_CONFIDENCE = 0.45
LOW_CONFIDENCE = 0.30


class TCodeSearchService:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load knowledge base
        self.tcodes = load_knowledge()
        self.texts = [t["text"] for t in self.tcodes]

        # Build embeddings
        self.embeddings = self.model.encode(self.texts, convert_to_numpy=True)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Create FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print(f"Semantic search initialized successfully: rows={len(self.tcodes)}, dim={dim}")

    # --------------------------------------------------
    # Main Search Function
    # --------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> Dict:

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            item = self.tcodes[idx]
            results.append({
                "tcode": item["tcode"],
                "description": item["description"],
                "module": item["module"],
                "score": float(score)
            })

        best_score = results[0]["score"]

        # --------------------------------------------------
        # Confidence Intelligence (the important logic)
        # --------------------------------------------------

        # HIGH CONFIDENCE → direct answer
        if best_score >= HIGH_CONFIDENCE:
            return {
                "type": "direct",
                "best_match": results[0],
                "alternatives": results[1:3]
            }

        # MEDIUM CONFIDENCE → suggest + confirm
        if best_score >= MEDIUM_CONFIDENCE:
            return {
                "type": "suggestion",
                "message": f"I found a likely SAP transaction: {results[0]['tcode']} — {results[0]['description']}. Please confirm.",
                "best_match": results[0],
                "alternatives": results[1:4]
            }

        # LOW CONFIDENCE → clarification required
        if best_score >= LOW_CONFIDENCE:
            return {
                "type": "clarification",
                "message": "I need a bit more detail to identify the correct SAP transaction.",
                "examples": [
                    "create purchase requisition",
                    "post vendor invoice",
                    "display purchase order"
                ],
                "results": results
            }

        # VERY LOW CONFIDENCE → unrelated query
        return {
            "type": "no_match",
            "message": "This doesn't appear to be an SAP transaction request. Please describe the SAP task you want to perform."
        }


# Singleton instance
search_service = TCodeSearchService()