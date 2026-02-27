# app/services/search_service.py
"""
Lightweight, defensive semantic search service for SAP T-codes.

- Uses embed_query() and load_knowledge() when available.
- Defensive imports and clear error logging so it will not crash Render.
- Exports:
    initialize_search()  # run in background on startup
    is_ready()
    search_tcode(query, top_k)
    _get_index_meta()    # debug support
"""

from typing import List, Dict, Optional, Any
import threading
import traceback
import time
import numpy as np

# runtime globals
_index: Optional[np.ndarray] = None
_metadata: Optional[List[Dict]] = None
_ready: bool = False
_lock = threading.Lock()
_last_init_error: Optional[str] = None
_last_init_time: Optional[float] = None


def is_ready() -> bool:
    return bool(_ready)


def _get_index_meta() -> Dict[str, Any]:
    try:
        rows = int(_index.shape[0]) if _index is not None else 0
        dim = int(_index.shape[1]) if _index is not None else 0
        meta_len = len(_metadata) if _metadata else 0
        return {"rows": rows, "dim": dim, "metadata_len": meta_len, "ready": _ready, "last_error": _last_init_error}
    except Exception:
        return {"rows": 0, "dim": 0, "metadata_len": 0, "ready": _ready, "last_error": _last_init_error}


def initialize_search() -> None:
    """
    Safe initialization: attempts to import loader + embedder and build normalized index.
    If anything is missing or fails, it records the error and leaves _ready = False.
    """
    global _index, _metadata, _ready, _last_init_error, _last_init_time

    with _lock:
        if _ready:
            print("Search already initialized — skipping")
            return

        _last_init_time = time.time()
        print("🔄 Initializing semantic search engine... (defensive)")

        try:
            # dynamic imports to surface import problems here
            from app.services.knowledge_loader import load_knowledge
            from app.services.embedding_service import embed_query
        except Exception as imp_exc:
            _last_init_error = f"Import failure: {imp_exc}"
            print("❌ Failed to initialize search - import error:", _last_init_error)
            traceback.print_exc()
            _ready = False
            return

        if not callable(load_knowledge):
            _last_init_error = "load_knowledge is not callable or missing"
            print("❌", _last_init_error)
            _ready = False
            return

        if not callable(embed_query):
            _last_init_error = "embed_query is not callable or missing"
            print("❌", _last_init_error)
            _ready = False
            return

        try:
            loaded = load_knowledge()
            if not isinstance(loaded, tuple) or len(loaded) < 2:
                raise RuntimeError("load_knowledge must return (embeddings, metadata)")

            index_raw, metadata = loaded[0], loaded[1]

            if index_raw is None:
                raise RuntimeError("Embeddings index is None")

            index_arr = np.array(index_raw, dtype=np.float32)
            if index_arr.ndim != 2:
                raise RuntimeError(f"Embeddings must be 2-D, got {index_arr.shape}")

            # normalize
            norms = np.linalg.norm(index_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            index_norm = index_arr / norms

            _index = index_norm
            _metadata = metadata or []
            _ready = True
            _last_init_error = None
            print(f"✅ Semantic search initialized: rows={_index.shape[0]}, dim={_index.shape[1]}")

        except Exception as exc:
            _last_init_error = f"runtime failure: {exc}"
            _ready = False
            print("❌ Failed to initialize search:", _last_init_error)
            traceback.print_exc()
            return


def _normalize_query(q_emb: np.ndarray) -> np.ndarray:
    v = q_emb.astype(np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / (n + 1e-10)


def _scores_to_results(scores: np.ndarray, top_idx: List[int]) -> List[Dict]:
    res = []
    for i in top_idx:
        item = _metadata[i] if _metadata and i < len(_metadata) else {}
        res.append({
            "tcode": item.get("tcode"),
            "description": item.get("description"),
            "module": item.get("module"),
            "score": float(scores[i])
        })
    return res


def search_tcode(query: str, top_k: int = 5) -> Any:
    """
    Defensive search endpoint: returns warming_up or error dicts if initialization failed.
    """
    if not _ready or _index is None:
        return {"status": "warming_up", "message": "AI knowledge base not ready", "last_error": _last_init_error}

    if not query or not query.strip():
        return {"status": "invalid_query", "message": "Empty query"}

    try:
        from app.services.embedding_service import embed_query
    except Exception as e:
        return {"status": "embedding_missing", "message": str(e)}

    try:
        q_emb = embed_query(query)
        if q_emb is None:
            return {"status": "embedding_error"}

        q_arr = np.array(q_emb, dtype=np.float32)
        if q_arr.ndim > 1:
            q_arr = q_arr.reshape(-1)
        if q_arr.shape[0] != _index.shape[1]:
            return {"status": "shape_mismatch", "message": "Embedding dim mismatch"}

        q_vec = _normalize_query(q_arr)
        scores = np.dot(_index, q_vec).astype(np.float32)

        top_k = max(1, min(int(top_k), len(scores)))
        top_idx = np.argsort(scores)[-top_k:][::-1]
        results = _scores_to_results(scores, top_idx.tolist())

        top_score = results[0]["score"] if results else 0.0
        if top_score >= 0.75:
            return {"type": "confident", "best_match": results[0], "alternatives": results[1:3], "results": results}
        elif top_score >= 0.55:
            return {"type": "suggestion", "options": results[:3], "results": results}
        else:
            return {"type": "clarification", "message": "Need more detail", "results": results}

    except Exception as exc:
        print("Search runtime error:", str(exc))
        traceback.print_exc()
        return {"status": "error", "message": str(exc)}