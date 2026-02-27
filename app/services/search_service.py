# app/services/search_service.py
"""
Production-ready semantic search service for SAP T-codes with confidence tiers.

Features:
- Lazy initialization via initialize_search() (safe for Render/health checks)
- Robust normalization for cosine similarity
- Defenses for shape mismatches and missing data
- Diagnostic logging & debug file when initialization fails
- _get_index_meta() debug accessor
- Backwards-compatible alias search_tcode()
- Confidence tiers: "confident", "suggestion", "clarification"
"""

from typing import List, Dict, Optional, Tuple, Any
import threading
import traceback
import os

import numpy as np

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

def _get_index_meta() -> Dict[str, Any]:
    """Return simple metadata about the index for debug endpoints."""
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
    Includes diagnostics and writes a debug file on failure.
    """
    global _index, _metadata, _ready

    if _ready:
        print("Search already initialized — skipping")
        return

    with _lock:
        if _ready:
            return

        try:
            print("🔄 Initializing semantic search engine... (debug mode)")

            # Diagnostics: verify loader / embed function presence
            print(f"DEBUG: load_knowledge is {load_knowledge!r}")
            print(f"DEBUG: embed_query is {embed_query!r}")

            if not callable(load_knowledge):
                raise RuntimeError("load_knowledge is not callable (None or wrong import)")

            if not callable(embed_query):
                raise RuntimeError("embed_query is not callable (None or wrong import)")

            print("Starting background knowledge initialization...")
            loaded = load_knowledge()

            # Support multiple return shapes from load_knowledge()
            if isinstance(loaded, tuple):
                if len(loaded) >= 2:
                    index_raw = loaded[0]
                    metadata = loaded[1]
                else:
                    raise RuntimeError("load_knowledge returned tuple with insufficient elements")
            else:
                raise RuntimeError("load_knowledge did not return a tuple (index, metadata)")

            print(f"DEBUG: load_knowledge returned index type={type(index_raw)}, metadata type={type(metadata)}")

            if index_raw is None:
                raise RuntimeError("Embeddings index (index_raw) is None")

            # Convert to numpy array and validate shape
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

            print(f"✅ Semantic search initialized successfully: rows={n_rows}, dim={dim}, metadata_len={len(_metadata)}")

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
# Search helpers & main public functions
# ------------------------
def _normalize_query_vector(q_emb: np.ndarray) -> np.ndarray:
    """Normalize a 1-D query embedding vector safely."""
    q = q_emb.astype(np.float32).reshape(-1)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        q_norm = 1.0
    return q / (q_norm + 1e-10)

def _scores_to_results(scores: np.ndarray, top_indices: np.ndarray, top_k: int) -> List[Dict]:
    """Build result dicts from scores and metadata."""
    global _metadata
    results: List[Dict] = []
    for idx in top_indices:
        if idx < 0 or idx >= len(_metadata):
            continue
        item = _metadata[idx] or {}
        results.append({
            "tcode": item.get("tcode"),
            "description": item.get("description"),
            "module": item.get("module"),
            "score": round(float(scores[idx]), 6)
        })
    return results

# Backwards-compatible name used by main.py
def search(query: str, top_k: int = 5) -> Any:
    return search_tcode(query, top_k)

def search_tcode(query: str, top_k: int = 5) -> Any:
    """
    Perform semantic search for the given query and return results with confidence tiers.

    Return structure (dict):
      - type: "confident" | "suggestion" | "clarification"
      - results: list of result dicts (always present for compatibility)
      - additional fields differ by type:
          confident: best_match, alternatives
          suggestion: message, options
          clarification: message, examples
    """
    global _index, _metadata, _ready

    # Warmup handling - do not crash; allow health checks to pass while background init runs
    if not _ready or _index is None or _metadata is None:
        return {
            "status": "warming_up",
            "message": "AI knowledge base loading (~20s first deploy)"
        }

    if not query or not query.strip():
        return {
            "status": "invalid_query",
            "message": "Query cannot be empty."
        }

    try:
        # Embed the query
        q_emb_raw = embed_query(query)
        if q_emb_raw is None:
            return {"status": "embedding_error", "message": "Failed to embed query."}

        q_emb = np.array(q_emb_raw, dtype=np.float32)
        if q_emb.ndim > 1:
            q_emb = q_emb.reshape(-1)
        elif q_emb.ndim == 0:
            return {"status": "embedding_error", "message": "Invalid embedding shape."}

        # Dimension check
        if _index is None or q_emb.shape[0] != _index.shape[1]:
            return {
                "status": "shape_mismatch",
                "message": f"Embedding dimension ({q_emb.shape[0]}) does not match index dimension ({_index.shape[1] if _index is not None else 'None'})."
            }

        # Normalize query vector
        q_vec = _normalize_query_vector(q_emb)

        # Cosine similarity
        scores = np.dot(_index, q_vec)  # shape (N,)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        n_items = len(_metadata)
        if n_items == 0:
            return {"status": "no_data", "message": "No indexed data available."}

        top_k = max(1, int(top_k))
        top_k = min(top_k, n_items)

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = _scores_to_results(scores, top_indices, top_k)

        if not results:
            return {"status": "no_results", "message": "No matching results."}

        # ---- Confidence Intelligence ----
        top_score = results[0]["score"]

        # Keep 'results' always for backward compatibility, plus a typed response
        if top_score >= 0.75:
            return {
                "type": "confident",
                "best_match": results[0],
                "alternatives": results[1:3],
                "results": results
            }
        elif top_score >= 0.55:
            return {
                "type": "suggestion",
                "message": "I think you might mean one of these:",
                "options": results[:3],
                "results": results
            }
        else:
            return {
                "type": "clarification",
                "message": "I need a bit more detail to identify the correct SAP transaction. Examples:",
                "examples": [
                    "create purchase requisition",
                    "post vendor invoice",
                    "display purchase order"
                ],
                "results": results
            }

    except Exception as exc:
        # Catch-all to avoid worker crash
        print("Search runtime error:", str(exc))
        traceback.print_exc()
        return {"status": "error", "message": f"Search encountered an error: {str(exc)}"}

# ------------------------
# Backwards compatibility alias (explicit)
# ------------------------
def search_tcode_alias(query: str, top_k: int = 5):
    return search_tcode(query, top_k)