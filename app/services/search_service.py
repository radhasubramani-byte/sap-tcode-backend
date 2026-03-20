"""
Production-safe semantic search service for SAP T-codes.

Exposes:
- initialize_search()  -> lazy initialization (call on startup in background)
- is_ready()           -> bool (if embeddings/index ready)
- search_tcode(query, top_k=5) -> returns a dict with confidence and results
- search(query, top_k=5) -> compatibility alias
"""

import os
import threading
from typing import List, Dict, Optional, Tuple
import numpy as np
import glob
import csv
import re

try:
    from app.services.knowledge_loader import load_knowledge
except Exception:
    load_knowledge = None

try:
    from app.services.embedding_service import embed_query
except Exception:
    embed_query = None


_index: Optional[np.ndarray] = None
_metadata: Optional[List[Dict]] = None
_ready: bool = False
_init_started: bool = False
_index_lock = threading.Lock()

_alias_patterns: List[Dict] = []
_tcode_index_map: Dict[str, Dict] = {}

_EMBED_PATH = "/app/data/embeddings.npy" if os.path.isdir("/app/data") else "data/embeddings.npy"
_DATA_DIR = "/app/data" if os.path.isdir("/app/data") else "data"


def is_ready() -> bool:
    return _ready


def _normalize_embeddings(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _safe_load_embeddings() -> Optional[np.ndarray]:
    try:
        if os.path.exists(_EMBED_PATH):
            arr = np.load(_EMBED_PATH)
            if arr is not None and len(arr) > 0:
                return arr
    except Exception as exc:
        print(f"Failed to load cached embeddings: {exc}")
    return None


def _safe_save_embeddings(arr: np.ndarray) -> None:
    try:
        os.makedirs(os.path.dirname(_EMBED_PATH), exist_ok=True)
        np.save(_EMBED_PATH, arr)
    except Exception as exc:
        print(f"Failed to save embeddings: {exc}")


def _normalize_text_for_match(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_alias_patterns() -> None:
    global _alias_patterns
    _alias_patterns = []

    if not os.path.isdir(_DATA_DIR):
        print(f"Alias data directory not found: {_DATA_DIR}")
        return

    pattern = os.path.join(_DATA_DIR, "*_aliases.csv")
    files = sorted(glob.glob(pattern))

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
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

                    _alias_patterns.append(
                        {
                            "alias": _normalize_text_for_match(alias),
                            "raw_alias": alias,
                            "tcode": tcode,
                            "canonical_desc": canonical,
                            "source": fname,
                        }
                    )
                    count += 1

                print(f"Loaded {count} aliases from {fname}")
        except Exception as exc:
            print(f"Failed to load alias file {fname}: {exc}")

    if _alias_patterns:
        print("Alias patterns initialized")
    else:
        print("No alias files found")


def _find_alias_matches(query: str) -> List[Dict]:
    if not query or not _alias_patterns:
        return []

    q = _normalize_text_for_match(query)
    matches: List[Dict] = []

    for entry in _alias_patterns:
        alias = entry.get("alias", "")
        if not alias:
            continue

        if q == alias:
            matches.append(
                {
                    "tcode": entry.get("tcode"),
                    "description": entry.get("canonical_desc") or entry.get("raw_alias"),
                    "matched_alias": entry.get("raw_alias"),
                    "source": entry.get("source"),
                    "score": 1.0,
                }
            )
            continue

        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, q):
            matches.append(
                {
                    "tcode": entry.get("tcode"),
                    "description": entry.get("canonical_desc") or entry.get("raw_alias"),
                    "matched_alias": entry.get("raw_alias"),
                    "source": entry.get("source"),
                    "score": 0.95,
                }
            )
            continue

        if alias in q:
            matches.append(
                {
                    "tcode": entry.get("tcode"),
                    "description": entry.get("canonical_desc") or entry.get("raw_alias"),
                    "matched_alias": entry.get("raw_alias"),
                    "source": entry.get("source"),
                    "score": 0.8,
                }
            )

    matches.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return matches


def initialize_search() -> None:
    global _index, _metadata, _ready, _tcode_index_map, _init_started

    with _index_lock:
        if _ready:
            print("Search already initialized — skipping")
            return

        if _init_started:
            print("Search initialization already started — skipping duplicate call")
            return

        _init_started = True

    print("Initializing semantic search engine...")

    try:
        load_alias_patterns()

        if not callable(load_knowledge):
            raise RuntimeError("load_knowledge() not found")

        loaded = load_knowledge()
        index_raw = None
        metadata = None

        if isinstance(loaded, tuple) and len(loaded) == 2:
            index_raw, metadata = loaded
        elif isinstance(loaded, list):
            metadata = loaded
        elif loaded is None:
            metadata = []
        else:
            metadata = loaded if isinstance(loaded, list) else []

        if index_raw is not None and isinstance(index_raw, np.ndarray) and index_raw.size > 0:
            _index = _normalize_embeddings(index_raw)
        else:
            arr = _safe_load_embeddings()
            if arr is not None:
                _index = _normalize_embeddings(arr)
            else:
                _index = None

        _metadata = metadata or []

        _tcode_index_map = {}
        for row in _metadata:
            if isinstance(row, dict):
                t = row.get("tcode")
                if t:
                    _tcode_index_map[str(t).upper()] = row

        if _index is None and _metadata and callable(embed_query):
            texts = []
            for row in _metadata:
                if not isinstance(row, dict):
                    texts.append("")
                    continue
                text = row.get("text") or f"{row.get('description', '')} {row.get('module', '')}"
                texts.append(text)

            try:
                emb_arr = embed_query(texts)  # type: ignore[arg-type]
            except TypeError:
                embs = []
                for text in texts:
                    e = embed_query(text)
                    embs.append(np.array(e))
                emb_arr = np.vstack(embs)

            emb_arr = np.array(emb_arr, dtype=np.float32)
            emb_arr = _normalize_embeddings(emb_arr)
            _index = emb_arr
            _safe_save_embeddings(emb_arr)
            print(f"Saved embeddings cache to {_EMBED_PATH}")

        if _index is None or not isinstance(_index, np.ndarray) or len(_index) == 0:
            print("No embeddings available. Search not ready.")
            _ready = False
        else:
            _ready = True
            print(
                f"Semantic search initialized: rows={len(_metadata)}, "
                f"dim={_index.shape[1] if _index is not None else 'NA'}"
            )

    except Exception as exc:
        print(f"Failed to initialize search: {exc}")
        _ready = False


def build_response_with_confidence(results: List[Dict]) -> Dict:
    if not results:
        return {
            "type": "none",
            "best_match": None,
            "alternatives": [],
            "results": [],
            "confidence": 0.0,
            "confidence_label": "low",
        }

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
        "confidence_label": label,
    }


def _cosine_search(q_emb: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    global _index
    if _index is None or _index.size == 0:
        return []

    q_emb = np.array(q_emb, dtype=np.float32)
    denom = np.linalg.norm(q_emb) + 1e-10
    qv = q_emb / denom
    scores = np.dot(_index, qv)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(int(i), float(scores[i])) for i in top_indices]


def search_tcode(query: str, top_k: int = 5) -> Dict:
    global _index, _metadata, _ready, _tcode_index_map

    if not query or not query.strip():
        return {"status": "invalid_query", "message": "Query cannot be empty"}

    alias_matches = _find_alias_matches(query)
    if alias_matches:
        results = []
        for m in alias_matches[:top_k]:
            tcode = (m.get("tcode") or "").upper() if m.get("tcode") else None
            meta = _tcode_index_map.get(tcode, {}) if tcode else {}
            desc = meta.get("description") or m.get("description")
            module = meta.get("module") or meta.get("module_name") or None

            results.append(
                {
                    "tcode": tcode,
                    "description": desc,
                    "module": module,
                    "score": round(float(m.get("score", 0.9)), 6),
                }
            )
        return build_response_with_confidence(results)

    if not _ready or _index is None:
        return {
            "status": "warming_up",
            "message": "AI knowledge base loading (~first deploy may take longer)",
        }

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
            results.append(
                {
                    "tcode": item.get("tcode") if isinstance(item, dict) else None,
                    "description": item.get("description") if isinstance(item, dict) else None,
                    "module": item.get("module") if isinstance(item, dict) else None,
                    "score": round(float(score), 6),
                }
            )

        return build_response_with_confidence(results)

    except Exception as exc:
        print(f"Search runtime error: {exc}")
        return {"status": "error", "message": f"Search failed: {str(exc)}"}


def search(query: str, top_k: int = 5) -> Dict:
    return search_tcode(query, top_k=top_k)


__all__ = ["initialize_search", "is_ready", "search_tcode", "search"]