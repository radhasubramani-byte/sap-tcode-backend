import numpy as np
from typing import List, Dict, Tuple, Optional
from app.services.knowledge_loader import load_knowledge
from app.services.embedding_service import embed_query

# =========================================================
# Runtime state
# =========================================================
_index: Optional[np.ndarray] = None
_metadata: Optional[List[Dict]] = None
_ready: bool = False


# =========================================================
# Status check
# =========================================================
def is_ready() -> bool:
    return _ready


def _get_index_meta():
    if _index is None:
        return {"rows": 0, "dim": 0, "metadata_len": 0}
    return {
        "rows": int(_index.shape[0]),
        "dim": int(_index.shape[1]),
        "metadata_len": len(_metadata) if _metadata else 0,
    }


# =========================================================
# Initialization
# =========================================================
def initialize_search():
    global _index, _metadata, _ready

    if _ready:
        print("Search already initialized — skipping")
        return

    print("🔄 Initializing semantic search engine...")

    try:
        loaded = load_knowledge()

        # ---- FIX: handle multiple return formats ----
        if isinstance(loaded, tuple):
            if len(loaded) == 2:
                index_raw, metadata = loaded
            elif len(loaded) >= 3:
                index_raw, metadata = loaded[0], loaded[1]
            else:
                raise Exception("Invalid load_knowledge return format")
        else:
            raise Exception("load_knowledge did not return tuple")

        if index_raw is None or len(index_raw) == 0:
            raise Exception("Embeddings index empty")

        # normalize vectors
        norms = np.linalg.norm(index_raw, axis=1, keepdims=True)
        norms[norms == 0] = 1
        index_raw = index_raw / norms

        _index = index_raw
        _metadata = metadata
        _ready = True

        print(f"✅ Semantic search initialized successfully: {_get_index_meta()}")

    except Exception as e:
        print("❌ Failed to initialize search:", str(e))
        _ready = False


# =========================================================
# Search
# =========================================================
def search(query: str, top_k: int = 5) -> List[Dict]:
    return search_tcode(query, top_k)


def search_tcode(query: str, top_k: int = 5) -> List[Dict]:

    if not _ready or _index is None:
        return [{
            "status": "warming_up",
            "message": "AI knowledge base loading (~20s first deploy)"
        }]

    if not query or not query.strip():
        return [{"status": "invalid_query"}]

    try:
        q_emb = embed_query(query)

        if q_emb is None or len(q_emb) == 0:
            return [{"status": "embedding_error"}]

        # normalize query
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)

        scores = np.dot(_index, q_emb)

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
        return [{"status": "error", "message": str(e)}]