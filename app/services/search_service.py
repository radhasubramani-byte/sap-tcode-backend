# app/services/search_service.py
"""
Production-safe semantic search service for SAP T-codes.

Exposes:
- initialize_search()  -> lazy initialization (call on startup in background)
- is_ready()           -> bool (if embeddings/index ready)
- search_tcode(query, top_k=5) -> returns a dict with confidence and results
- (also provides `search` as alias for compatibility)
"""

import os
import threading
from typing import List, Dict, Optional, Tuple
import numpy as np
import glob
import csv
import re

# Try to import loader & embedder. These modules should exist in your repo.
# If they do not, we'll keep placeholders and fail gracefully.
try:
    from app.services.knowledge_loader import load_knowledge
except Exception:
    load_knowledge = None

try:
    from app.services.embedding_service import embed_query
except Exception:
    embed_query = None

# -------------------------
# Global runtime state
# -------------------------
_index: Optional[np.ndarray] = None     # normalized embeddings (N x D)
_metadata: Optional[List[Dict]] = None  # list of records (dict)
_ready: bool = False
_index_lock = threading.Lock()

# Alias structures
_alias_patterns: List[Dict] = []        # list of {"alias": "create po", "tcode":"ME21N", "canonical_desc":"Create Purchase Order", "source":"mm_aliases.csv"}
_tcode_index_map: Dict[str, Dict] = {}  # tcode -> metadata dict for quick enrichment

# Cache path (where we write/read embeddings)
_EMBED_PATH = "/app/data/embeddings.npy" if os.path.isdir("/app/data") else "data/embeddings.npy"
_DATA_DIR = "/app/data" if os.path.isdir("/app/data") else "data"

# -------------------------
# Utilities
# -------------------------
def is_ready() -> bool:
    """Returns whether the semantic index is ready."""
    return _ready

def _normalize_embeddings(mat: np.ndarray) -> np.ndarray:
    """L2-normalize along rows (embedding vectors)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

def _safe_load_embeddings() -> Optional[np.ndarray]:
    """Attempt to load embeddings cache from disk."""
    try:
        if os.path.exists(_EMBED_PATH):
            arr = np.load(_EMBED_PATH)
            if arr is not None and len(arr) > 0:
                return arr
    except Exception:
        pass
    return None

def _safe_save_embeddings(arr: np.ndarray) -> None:
    try:
        os.makedirs(os.path.dirname(_EMBED_PATH), exist_ok=True)
        np.save(_EMBED_PATH, arr)
    except Exception:
        pass

# -------------------------
# Alias loader / matcher
# -------------------------
def _normalize_text_for_match(s: str) -> str:
    """Simple canonicalization for matching: lowercase, remove punctuation (keep spaces)."""
    if s is None:
        return ""
    s = s.lower()
    # replace non-alphanumeric with space
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_alias_patterns() -> None:
    """
    Load any *_aliases.csv files from the data directory.
    Expected CSV format: alias,tcode,canonical_desc
    Populates global _alias_patterns and prints what was loaded.
    """
    global _alias_patterns
    _alias_patterns = []
    if not os.path.isdir(_DATA_DIR):
        # nothing to load
        return

    pattern = os.path.join(_DATA_DIR, "*_aliases.csv")
    files = sorted(glob.glob(pattern))
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                # allow optional header
                rows = list(reader)
                start_idx = 0
                if rows and len(rows[0]) >= 1 and any("alias" in c.lower() for c in rows[0]):
                    start_idx = 1
                count = 0
                for row in rows[start_idx:]:
                    if not row or len(row) < 2:
                        continue
                    alias = row[0].strip()
                    tcode = row[1].strip() if len(row) > 1 else ""
                    canonical = row[2].strip() if len(row) > 2 else ""
                    if not alias:
                        continue
                    _alias_patterns.append({
                        "alias": _normalize_text_for_match(alias),
                        "raw_alias": alias,
                        "tcode": tcode,
                        "canonical_desc": canonical,
                        "source": fname
                    })
                    count += 1
            print(f"Loaded {count} aliases from {fname}")
        except Exception as e:
            print(f"Failed to load alias file {fname}: {repr(e)}")

    if _alias_patterns:
        # build an index by tcode if possible (done later in initialize_search)
        print("Alias patterns initialized")
    else:
        print("No alias files found (no *_aliases.csv in data dir)")

def _find_alias_matches(query: str) -> List[Dict]:
    """
    Return list of matches [{tcode, description, score_estimate, matched_alias, source}, ...]
    We use simple word-boundary substring checks (case-insensitive).
    """
    global _alias_patterns
    if not query or not _alias_patterns:
        return []

    q = _normalize_text_for_match(query)
    matches = []
    for entry in _alias_patterns:
        alias = entry.get("alias", "")
        if not alias:
            continue
        # exact equality
        if q == alias:
            matches.append({
                "tcode": entry.get("tcode"),
                "description": entry.get("canonical_desc") or entry.get("raw_alias"),
                "matched_alias": entry.get("raw_alias"),
                "source": entry.get("source"),
                "score": 1.0
            })
            continue
        # word-boundary search
        # escape alias for regex
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, q):
            matches.append({
                "tcode": entry.get("tcode"),
                "description": entry.get("canonical_desc") or entry.get("raw_alias"),
                "matched_alias": entry.get("raw_alias"),
                "source": entry.get("source"),
                "score": 0.95
            })
            continue
        # substring fallback (less preferred)
        if alias in q:
            matches.append({
                "tcode": entry.get("tcode"),
                "description": entry.get("canonical_desc") or entry.get("raw_alias"),
                "matched_alias": entry.get("raw_alias"),
                "source": entry.get("source"),
                "score": 0.8
            })
    # sort by score desc
    matches.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return matches

# -------------------------
# Knowledge initialization
# -------------------------
def initialize_search():
    """
    Load knowledge and embeddings lazily (safe to call multiple times).
    Intended to be launched in a background thread at startup (not at import time).
    """
    global _index, _metadata, _ready, _tcode_index_map

    with _index_lock:
        if _ready:
            print("Search already initialized — skipping")
            return

        print("🔄 Initializing semantic search engine... (defensive)")
        try:
            # load aliases first (so alias-based answers are available even if embeddings not ready)
            load_alias_patterns()

            if not callable(load_knowledge):
                raise RuntimeError("load_knowledge() not found")

            # load_knowledge returns (index_raw, metadata) OR (metadata only) depending on implementation
            loaded = load_knowledge()
            # Support both return shapes:
            index_raw = None
            metadata = None

            if isinstance(loaded, tuple) and len(loaded) == 2:
                index_raw, metadata = loaded
            elif isinstance(loaded, list):
                metadata = loaded
            else:
                if loaded is None:
                    metadata = []
                else:
                    metadata = loaded if isinstance(loaded, list) else []

            # If index raw available, normalize and set
            if index_raw is not None and isinstance(index_raw, np.ndarray) and index_raw.size > 0:
                index_raw = _normalize_embeddings(index_raw)
                _index = index_raw
            else:
                # Try to read precomputed embeddings from disk
                arr = _safe_load_embeddings()
                if arr is not None:
                    _index = _normalize_embeddings(arr)
                else:
                    # We'll generate embeddings at runtime below (requires embed_query)
                    _index = None

            _metadata = metadata or []

            # Build tcode index map for enrichment (useful for alias lookups)
            _tcode_index_map = {}
            for row in (_metadata or []):
                if not isinstance(row, dict):
                    continue
                t = row.get("tcode")
                if t:
                    _tcode_index_map[str(t).upper()] = row

            # If we don't have embeddings but there is metadata and embed_query exists, build now
            if _index is None and _metadata and callable(embed_query):
                # Build embeddings for all metadata rows by composing the text we want to embed
                texts = []
                for row in _metadata:
                    text = row.get("text") or f"{row.get('description','')} {row.get('module','')}"
                    texts.append(text)

                try:
                    emb_arr = embed_query(texts)  # type: ignore
                except TypeError:
                    embs = []
                    for t in texts:
                        e = embed_query(t)
                        embs.append(np.array(e))
                    emb_arr = np.vstack(embs)

                emb_arr = np.array(emb_arr, dtype=np.float32)
                emb_arr = _normalize_embeddings(emb_arr)
                _index = emb_arr
                _safe_save_embeddings(emb_arr)
                print(f"💾 Saved embeddings cache to {_EMBED_PATH}")

            # Final safety checks
            if _index is None or (isinstance(_index, np.ndarray) and len(_index) == 0):
                print("⚠️ No embeddings available yet — search will warm up and generate embeddings on demand (if possible)")
                _ready = False
            else:
                _ready = True
                print(f"✅ Semantic search initialized: rows={len(_metadata)}, dim={_index.shape[1] if _index is not None else 'NA'}")

        except Exception as e:
            print("❌ Failed to initialize search:", repr(e))
            _ready = False

# -------------------------
# Response builder (confidence intelligence)
# -------------------------
def build_response_with_confidence(results: List[Dict]) -> Dict:
    """
    Input:
        results: list of dicts already sorted in descending score order
    Returns:
        dict: {
            "type": "confident"|"uncertain"|"none",
            "best_match": {...} or None,
            "alternatives": [...],
            "results": [...],
            "confidence": float (0..1),
            "confidence_label": "high"|"medium"|"low"
        }
    """
    if not results:
        return {"type": "none", "best_match": None, "alternatives": [], "results": [], "confidence": 0.0, "confidence_label": "low"}

    best = results[0]
    try:
        max_score = float(best.get("score", 0.0))
    except Exception:
        max_score = 0.0

    if max_score >= 0.70:
        label = "high"
    elif max_score >= 0.40:
        label = "medium"
    else:
        label = "low"

    return {
        "type": "confident" if max_score >= 0.40 else "uncertain",
        "best_match": best,
        "alternatives": results[1:],
        "results": results,
        "confidence": round(max_score, 3),
        "confidence_label": label
    }

# -------------------------
# Main search function (public)
# -------------------------
def _cosine_search(q_emb: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    """Return list of (index, score) sorted desc by score."""
    global _index
    if _index is None or _index.size == 0:
        return []
    # ensure q_emb normalized
    q_emb = np.array(q_emb, dtype=np.float32)
    denom = np.linalg.norm(q_emb) + 1e-10
    qv = q_emb / denom
    scores = np.dot(_index, qv)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(int(i), float(scores[i])) for i in top_indices]

def search_tcode(query: str, top_k: int = 5) -> Dict:
    """
    Public search function expected by main.py.
    Returns a rich dict (see build_response_with_confidence).
    If the engine isn't ready it returns warming_up message as a dict.
    """
    global _index, _metadata, _ready, _tcode_index_map

    if not query or not query.strip():
        return {"status": "invalid_query", "message": "Query cannot be empty"}

    # Attempt alias match FIRST (works even if embeddings are not ready)
    alias_matches = _find_alias_matches(query)
    if alias_matches:
        # Build results enriched with metadata when available
        results = []
        for m in alias_matches[:top_k]:
            tcode = (m.get("tcode") or "").upper() if m.get("tcode") else None
            meta = _tcode_index_map.get(tcode, {}) if tcode else {}
            desc = meta.get("description") or m.get("description")
            module = meta.get("module") or meta.get("module_name") or None
            results.append({
                "tcode": tcode,
                "description": desc,
                "module": module,
                "score": round(float(m.get("score", 0.9)), 6)
            })
        return build_response_with_confidence(results)

    # If engine not ready, return warming message (we already attempted alias)
    if not _ready or _index is None:
        return {"status": "warming_up", "message": "AI knowledge base loading (~first deploy may take longer)"}

    # Embed query (defensive)
    if not callable(embed_query):
        return {"status": "error", "message": "Embedding service unavailable"}

    q_emb = embed_query(query)
    if q_emb is None:
        return {"status": "embedding_error", "message": "Failed to embed query"}

    try:
        matches = _cosine_search(q_emb, top_k=top_k)
        results = []
        for idx, score in matches:
            item = _metadata[idx] if (_metadata and idx < len(_metadata)) else {}
            results.append({
                "tcode": item.get("tcode") if isinstance(item, dict) else None,
                "description": item.get("description") if isinstance(item, dict) else None,
                "module": item.get("module") if isinstance(item, dict) else None,
                "score": round(float(score), 6)
            })
        # Build rich response with confidence intelligence
        response = build_response_with_confidence(results)
        return response
    except Exception as e:
        print("Search runtime error:", repr(e))
        return {"status": "error", "message": f"Search failed: {str(e)}"}

# Provide aliases for compatibility (main will look for several names)
def search(query: str, top_k: int = 5) -> Dict:
    return search_tcode(query, top_k=top_k)

# Exported API: initialize_search, is_ready, search_tcode, search
__all__ = ["initialize_search", "is_ready", "search_tcode", "search"]