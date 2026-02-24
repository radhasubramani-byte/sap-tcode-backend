import numpy as np
from typing import List, Dict

from app.services.knowledge_loader import load_knowledge
from app.services.embedding_service import embed_query


# =========================================================
# Global runtime state (DO NOT LOAD AT IMPORT TIME)
# =========================================================
_index = None
_metadata = None
_ready = False


# =========================================================
# Status Check
# =========================================================
def is_ready() -> bool:
    """
    Returns True when embeddings + knowledge base are ready.
    """
    return _ready


# =========================================================
# Lazy Initialization (called from main.py startup thread)
# =========================================================
def initialize_search():
    """
    Loads knowledge base and embeddings.
    MUST ONLY RUN AFTER FASTAPI STARTS.
    """
    global _index, _metadata, _ready

    if _ready:
        print("Search already initialized — skipping")
        return

    print("Starting knowledge loading inside initialize_search() ...")

    try:
        _index, _metadata = load_knowledge()
        _ready = True
        print(f"Knowledge fully loaded! Records: {len(_metadata)}")

    except Exception as e:
        print("❌ Failed to initialize search:", str(e))
        _ready = False


# =========================================================
# Search Function
# =========================================================
def search_tcode(query: str, top_k: int = 5) -> List[Dict]:
    """
    Semantic search for SAP T-codes.
    """

    # If still loading → don't crash server
    if not _ready or _index is None:
        return [{
            "status": "warming_up",
            "message": "Knowledge base is loading. Please retry in a few seconds."
        }]

    if not query or not query.strip():
        return [{
            "status": "invalid_query",
            "message": "Query cannot be empty"
        }]

    try:
        # Generate embedding
        q_emb = embed_query(query)

        # Cosine similarity via dot product (normalized embeddings assumed)
        scores = np.dot(_index, q_emb)

        # Get top matches
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            item = _metadata[idx]

            results.append({
                "tcode": item.get("tcode"),
                "description": item.get("description"),
                "score": float(scores[idx])
            })

        return results

    except Exception as e:
        return [{
            "status": "error",
            "message": f"Search failed: {str(e)}"
        }]