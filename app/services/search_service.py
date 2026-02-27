# app/services/search_service.py
"""
Production-safe semantic search service for SAP T-codes.

Features:
- Lazy initialization via initialize_search() (safe for Render/health checks)
- Robust normalization for cosine similarity
- Defenses for shape mismatches and missing data
- Diagnostic logging & debug file when initialization fails
- _get_index_meta() debug accessor
- Backwards-compatible alias search_tcode()

Expectations from other modules:
- load_knowledge() -> Tuple[Optional[list|np.ndarray], Optional[List[Dict]]]
    - embeddings_matrix: 2-D list or np.ndarray shaped (N, D)
    - metadata_list: list of dicts length N
- embed_query(query: str) -> 1-D list or np.ndarray of length D
"""

from typing import List, Dict, Optional, Tuple
import threading
import numpy as np
import traceback
import os

from app.services.knowledge_loader import load_knowledge
from app.services.embedding_service import embed_query

# ------------------------
# Global runtime state
# ------------------------
_index: Optional[np.ndarray] = None    # normalized embeddings matrix shape (N, D)
_metadata: Optional[List[Dict]] = None
_ready: bool = False
_lock = threading.Lock()

# ------------------------
# Health / readiness
# ------------------------
def is_ready() -> bool:
    """Return True when embeddings are loaded and ready for search."""
    return _ready

def _get_index_meta() -> Dict:
    """
    Debug accessor returning basic index metadata.
    Safe to call from other modules for health-checking/debug endpoints.
    """
    global _index, _metadata, _ready
    try:
        rows = int(_index.shape[0]) if (_index is not None and hasattr(_index, "shape")) else 0
        dim = int(_index.shape[1]) if (_index is not None and hasattr(_index, "shape")) else 0
    except Exception:
        rows, dim = 0, 0
    meta_len = len(_metadata) if _metadata is not None else 0
    return {"rows": rows, "dim": dim, "metadata_len": meta_len, "ready": _ready}

# ------------------------
# Initialization
# ------------------------
def initialize_search() -> None:
    """
    Lazy-load the knowledge and embeddings after the process starts.
    Designed to be called from a startup thread (so it doesn't block health checks).
    This version includes diagnostics and writes a debug file on failure.
    """
    global _index, _metadata, _ready

    # Fast return if already ready
    if _ready:
        print("Search already initialized — skipping")
        return

    with _lock:
        # Double-check after acquiring lock
        if _ready:
            return

        try:
            print("🔄 Initializing semantic search engine... (debug mode)")

            # DIAGNOSTIC: ensure loader / embedding funcs are actually callable
            print(f"DEBUG: load_knowledge is {load_knowledge!r}")
            print(f"DEBUG: embed_query is {embed_query!r}")

            if not callable(load_knowledge):
                raise RuntimeError("load_knowledge is not callable (None or wrong import)")

            if not callable(embed_query):
                raise RuntimeError("embed_query is not callable (None or wrong import)")

            print("Starting background knowledge initialization...")
            loaded: Tuple[Optional[np.ndarray], Optional[List[Dict]]] = load_knowledge()

            # Accept both tuple-return or other shapes; normalize below
            if not loaded:
                raise RuntimeError("load_knowledge returned no data or returned falsy value")

            index_raw, metadata = loaded

            print(f"DEBUG: load_knowledge returned index type={type(index_raw)}, metadata type={type(metadata)}")

            # Validate metadata
            if metadata is None or len(metadata) == 0:
                # allow metadata empty if index present, but fail loudly
                print("WARNING: metadata is None or empty from load_knowledge()")

            # Convert index to numpy array safely
            if index_raw is None:
                raise RuntimeError("Embeddings index (index_raw) is None")

            index_arr = np.array(index_raw, dtype=np.float32)

            if index_arr.ndim != 2:
                raise RuntimeError(f"Embeddings must be 2-D array, got shape {index_arr.shape}")

            n_rows, dim = index_arr.shape
            if n_rows == 0:
                raise RuntimeError("Embeddings index has zero rows")

            # Defensive: align metadata length with embedding rows
            if metadata is None:
                metadata = [{} for _ in range(n_rows)]
            elif len(metadata) != n_rows:
                print(
                    f"Warning: metadata length ({len(metadata)}) != embeddings rows ({n_rows}). "
                    "Truncating/expanding metadata to match embeddings."
                )
                if len(metadata) > n_rows:
                    metadata = metadata[:n_rows]
                else:
                    metadata = metadata + [{} for _ in range(n_rows - len(metadata))]

            # Normalize embeddings for stable cosine similarity
            norms = np.linalg.norm(index_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            index_normalized = index_arr / norms

            # Assign to globals only after full success
            _index = index_normalized.astype(np.float32)
            _metadata = metadata
            _ready = True

            print(f"✅ Semantic search initialized successfully: {n_rows} vectors, dim={dim}")

        except Exception as exc:
            _ready = False
            print("❌ Failed to initialize search:", str(exc))
            traceback.print_exc()
            # write a debug file for offline inspection
            try:
                debug_path = "/tmp/search_init_debug.log"
                with open(debug_path, "w", encoding="utf-8") as fh:
                    fh.write("Initialize search failed:\n")
                    fh.write(traceback.format_exc())
                print(f"Wrote initialization traceback to {debug_path}")
            except Exception as e:
                print("Failed to write /tmp/search_init_debug.log:", str(e))

# ------------------------
# Main search function
# ------------------------
def search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Perform semantic search for the given query.

    Return semantics:
      - If warming up or not ready: a single-entry list with status "warming_up"
      - If invalid query: a single-entry list with status "invalid_query"
      - On success: list of {tcode, description, score}
      - On failure: a single-entry list with status "error"
    """
    global _index, _metadata, _ready

    # Warmup handling - do not crash; allow health checks to pass while background init runs
    if not _ready or _index is None or _metadata is None:
        return [{
            "status": "warming_up",
            "message": "AI knowledge base loading (~20s first deploy)"
        }]

    if not query or not query.strip():
        return [{
            "status": "invalid_query",
            "message": "Query cannot be empty."
        }]

    try:
        # Embed the query
        q_emb_raw = embed_query(query)

        if q_emb_raw is None:
            return [{
                "status": "embedding_error",
                "message": "Failed to embed query."
            }]

        q_emb = np.array(q_emb_raw, dtype=np.float32)

        # Flatten to 1-D if necessary
        if q_emb.ndim > 1:
            q_emb = q_emb.reshape(-1)
        elif q_emb.ndim == 0:
            # scalar? invalid
            return [{
                "status": "embedding_error",
                "message": "Invalid embedding shape."
            }]

        # Safety: ensure dimensions match
        if _index is None or q_emb.shape[0] != _index.shape[1]:
            return [{
                "status": "shape_mismatch",
                "message": f"Embedding dimension ({q_emb.shape[0]}) does not match index dimension ({_index.shape[1] if _index is not None else 'None'})."
            }]

        # Normalize query vector (stable)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            q_norm = 1.0
        q_emb = q_emb / (q_norm + 1e-10)
        q_emb = q_emb.astype(np.float32)

        # Cosine similarity = dot(normalized_index, normalized_query)
        # _index is normalized on load
        scores = np.dot(_index, q_emb)  # shape (N,)

        # Defensive: ensure scores is 1-D numeric array
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        # Determine top_k safely
        n_items = len(_metadata)
        if n_items == 0:
            return [{
                "status": "no_data",
                "message": "No indexed data available."
            }]

        top_k = max(1, int(top_k))
        top_k = min(top_k, n_items)

        # argsort descending
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results: List[Dict] = []
        for idx in top_indices:
            # defensive index bounds check
            if idx < 0 or idx >= n_items:
                continue
            item = _metadata[idx] or {}
            results.append({
                "tcode": item.get("tcode"),
                "description": item.get("description"),
                "score": float(round(float(scores[idx]), 6))
            })

        return results

    except Exception as exc:
        # Catch-all to avoid worker crash
        print("Search runtime error:", str(exc))
        traceback.print_exc()
        return [{
            "status": "error",
            "message": f"Search encountered an error: {str(exc)}"
        }]

# ------------------------
# Backwards compatibility alias
# ------------------------
def search_tcode(query: str, top_k: int = 5):
    """
    Backwards-compatible wrapper for older imports expecting `search_tcode`.
    """
    return search(query, top_k)