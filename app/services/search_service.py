import numpy as np
from typing import List, Dict

load_knowledge = None
embed_query = None


# =========================================================
# Global runtime state (DO NOT LOAD AT IMPORT TIME)
# =========================================================
_index = None          # numpy matrix of embeddings
_metadata = None       # list of dict rows
_ready = False         # indicates embeddings finished loading


# =========================================================
# Status Check (used by /health)
# =========================================================
def is_ready() -> bool:
    return _ready


# =========================================================
# Lazy Initialization (called from main.py startup thread)
# =========================================================
def initialize_search():
    """
        global _index, _metadata, _ready, load_knowledge, embed_query

    from app.services.knowledge_loader import load_knowledge
    from app.services.embedding_service import embed_query

    """
    global _index, _metadata, _ready

    if _ready:
        print("Search already initialized — skipping")
        return

    print("🔄 Initializing semantic search engine...")

    try:
        index, metadata = load_knowledge()

        # Safety checks
        if index is None or len(index) == 0:
            raise Exception("Embeddings index empty")

        # Normalize embeddings (cosine similarity safety)
        norms = np.linalg.norm(index, axis=1, keepdims=True)
        norms[norms == 0] = 1
        index = index / norms

        _index = index
        _metadata = metadata
        _ready = True

        print(f"✅ Knowledge loaded successfully: {len(_metadata)} records")

    except Exception as e:
        print("❌ Failed to initialize search:", str(e))
        _ready = False


# =========================================================
# MAIN SEARCH FUNCTION (FastAPI uses THIS name)
# =========================================================
def search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Semantic SAP T-code search endpoint logic
    This is what main.py imports
    """

    # ---------- Warmup handling ----------
    if not _ready or _index is None:
        return [{
            "status": "warming_up",
            "message": "AI is starting — try again in 5 seconds"
        }]

    if not query or not query.strip():
        return [{
            "status": "invalid_query",
            "message": "Query cannot be empty"
        }]

    try:
        # ---------- Embed user query ----------
        q_emb = embed_query(query)

        if q_emb is None or len(q_emb) == 0:
            return [{
                "status": "embedding_error",
                "message": "Failed to embed query"
            }]

        # Normalize query vector
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)

        # ---------- Cosine similarity ----------
        scores = np.dot(_index, q_emb)

        # ---------- Top matches ----------
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            item = _metadata[idx]

            results.append({
                "tcode": item.get("tcode"),
                "description": item.get("description"),
                "score": round(float(scores[idx]), 4)
            })

        return results

    except Exception as e:
        print("Search runtime error:", str(e))
        return [{
            "status": "error",
            "message": f"Search failed: {str(e)}"
        }]