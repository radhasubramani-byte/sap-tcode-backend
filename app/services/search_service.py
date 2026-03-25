"""
search_service.py

Production-safe semantic search service for SAP T-codes.

Corrected search priority:
1. Alias exact match
2. Alias all-words match
3. Exact / lexical description match from metadata
4. Semantic search fallback

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


# -------------------------
# Global runtime state
# -------------------------
_index: Optional[np.ndarray] = None
_metadata: Optional[List[Dict]] = None
_ready: bool = False
_init_started: bool = False
_index_lock = threading.Lock()

_alias_patterns: List[Dict] = []
_tcode_index_map: Dict[str, Dict] = {}

# Prefer the actual Render path that your logs showed, then fall back
if os.path.isdir("/app/app/data"):
    _DATA_DIR = "/app/app/data"
    _EMBED_PATH = "/app/app/data/embeddings.npy"
elif os.path.isdir("/app/data"):
    _DATA_DIR = "/app/data"
    _EMBED_PATH = "/app/data/embeddings.npy"
else:
    _DATA_DIR = "data"
    _EMBED_PATH = "data/embeddings.npy"

STOP_WORDS = {
    "the", "a", "an", "for", "to", "in", "of", "on", "by", "with",
    "and", "or", "is", "are", "be", "do", "does", "did", "from"
}


# -------------------------
# Utilities
# -------------------------
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
    s = re.sub(r"[^a-z0-9/\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_meaningful(s: str) -> List[str]:
    tokens = _normalize_text_for_match(s).split()
    return [t for t in tokens if t and t not in STOP_WORDS]


# -------------------------
# Description matcher
# -------------------------
def _normalize_desc_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9/\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_description_matches(query: str, top_k: int = 5) -> List[Dict]:
    """
    Deterministic lexical matching against metadata descriptions.

    Priority:
    1) exact normalized description match
    2) query contained in description
    3) strong token overlap match
    """
    global _metadata

    if not query or not _metadata:
        return []

    q = _normalize_desc_text(query)
    if not q:
        return []

    q_tokens = set(q.split())

    exact_matches: List[Dict] = []
    contains_matches: List[Dict] = []
    overlap_matches: List[Dict] = []

    for row in _metadata:
        if not isinstance(row, dict):
            continue

        desc = (row.get("description") or "").strip()
        desc_norm = _normalize_desc_text(desc)
        if not desc_norm:
            continue

        tcode = row.get("tcode")
        module = row.get("module")

        # 1) Exact description match
        if desc_norm == q:
            exact_matches.append(
                {
                    "tcode": tcode,
                    "description": desc,
                    "module": module,
                    "score": 1.0,
                    "match_type": "description_exact",
                }
            )
            continue

        # 2) Contains match
        if q in desc_norm:
            contains_matches.append(
                {
                    "tcode": tcode,
                    "description": desc,
                    "module": module,
                    "score": 0.95,
                    "match_type": "description_contains",
                }
            )
            continue

        # 3) Token overlap match
        d_tokens = set(desc_norm.split())
        overlap = q_tokens.intersection(d_tokens)
        if overlap:
            ratio = len(overlap) / max(len(q_tokens), 1)
            if ratio >= 0.6:
                overlap_matches.append(
                    {
                        "tcode": tcode,
                        "description": desc,
                        "module": module,
                        "score": round(0.75 + (0.2 * ratio), 6),
                        "match_type": "description_overlap",
                    }
                )

    if exact_matches:
        return exact_matches[:top_k]

    if contains_matches:
        contains_matches.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return contains_matches[:top_k]

    overlap_matches.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return overlap_matches[:top_k]


# -------------------------
# Alias loader / matcher
# -------------------------
def load_alias_patterns() -> None:
    """
    Load any *_aliases.csv files from the data directory.
    Expected CSV format: alias,tcode,canonical_desc
    """
    global _alias_patterns
    _alias_patterns = []

    if not os.path.isdir(_DATA_DIR):
        print(f"Alias data directory not found: {_DATA_DIR}")
        return

    pattern = os.path.join(_DATA_DIR, "*_aliases.csv")
    files = sorted(glob.glob(pattern))

    for fpath in files:
        fname = os.path.basename(fpath)
        default_module = ""
        if fname.lower().startswith("mm_"):
            default_module = "MM"
        elif fname.lower().startswith("sd_"):
            default_module = "SD"
        elif fname.lower().startswith("le_"):
            default_module = "LE"

        try:
            with open(fpath, newline="", encoding="utf-8-sig") as fh:
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

                    if not alias or not tcode:
                        continue

                    alias_norm = _normalize_text_for_match(alias)
                    alias_tokens = _tokenize_meaningful(alias)

                    _alias_patterns.append(
                        {
                            "alias": alias_norm,
                            "alias_tokens": alias_tokens,
                            "raw_alias": alias,
                            "tcode": tcode.upper(),
                            "canonical_desc": canonical,
                            "module": default_module,
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


def _find_alias_exact_match(query: str) -> List[Dict]:
    if not query or not _alias_patterns:
        return []

    q = _normalize_text_for_match(query)
    matches: List[Dict] = []

    for entry in _alias_patterns:
        alias = entry.get("alias", "")
        if q == alias:
            matches.append(
                {
                    "tcode": entry.get("tcode"),
                    "description": entry.get("canonical_desc") or entry.get("raw_alias"),
                    "module": entry.get("module"),
                    "matched_alias": entry.get("raw_alias"),
                    "source": entry.get("source"),
                    "score": 1.0,
                    "match_type": "alias_exact",
                }
            )

    matches.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return matches


def _find_alias_all_words_match(query: str) -> List[Dict]:
    """
    Match alias rows where all meaningful query words are present in the alias.
    Example:
    query: "post goods receipt"
    alias: "post goods receipt" -> match
    alias: "goods receipt" -> no match because "post" is missing
    """
    if not query or not _alias_patterns:
        return []

    q_words = _tokenize_meaningful(query)
    if not q_words:
        return []

    q_set = set(q_words)
    matches: List[Dict] = []

    for entry in _alias_patterns:
        alias_words = set(entry.get("alias_tokens", []))
        if not alias_words:
            continue

        if all(word in alias_words for word in q_set):
            extra_words_penalty = len(alias_words) - len(q_set)
            score = 0.98 if extra_words_penalty == 0 else max(0.90, 0.98 - (0.01 * extra_words_penalty))
            matches.append(
                {
                    "tcode": entry.get("tcode"),
                    "description": entry.get("canonical_desc") or entry.get("raw_alias"),
                    "module": entry.get("module"),
                    "matched_alias": entry.get("raw_alias"),
                    "source": entry.get("source"),
                    "score": round(score, 6),
                    "match_type": "alias_all_words",
                    "_extra_words_penalty": extra_words_penalty,
                }
            )

    matches.sort(
        key=lambda x: (
            -x.get("score", 0.0),
            x.get("_extra_words_penalty", 999),
            x.get("tcode") or "",
        )
    )

    cleaned = []
    for m in matches:
        item = dict(m)
        item.pop("_extra_words_penalty", None)
        cleaned.append(item)

    return cleaned


# -------------------------
# Initialization
# -------------------------
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

        # Ready should reflect whether metadata is available for alias + lexical search,
        # not only whether semantic embeddings exist.
        if _metadata:
            _ready = True
            if isinstance(_index, np.ndarray) and len(_index) > 0:
                print(
                    f"Semantic search initialized: rows={len(_metadata)}, "
                    f"dim={_index.shape[1] if _index is not None else 'NA'}"
                )
            else:
                print(f"Lexical/alias search initialized without embeddings: rows={len(_metadata)}")
        else:
            print("No metadata available. Search not ready.")
            _ready = False

    except Exception as exc:
        print(f"Failed to initialize search: {exc}")
        _ready = False


# -------------------------
# Response builders
# -------------------------
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


def build_did_you_mean_response(results: List[Dict]) -> Dict:
    """
    Safe fallback when semantic confidence is too low.
    Returns suggestions instead of a possibly wrong best match.
    """
    cleaned_results = []
    for r in results[:3]:
        if isinstance(r, dict):
            cleaned_results.append(
                {
                    "tcode": r.get("tcode"),
                    "description": r.get("description"),
                    "module": r.get("module"),
                    "score": r.get("score"),
                    "match_type": r.get("match_type", "semantic"),
                }
            )

    return {
        "type": "uncertain",
        "message": "Did you mean one of these SAP T-codes?",
        "best_match": None,
        "alternatives": cleaned_results,
        "results": cleaned_results,
        "confidence": 0.0,
        "confidence_label": "low",
    }


# -------------------------
# Semantic search
# -------------------------
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


# -------------------------
# Public search function
# -------------------------
def search_tcode(query: str, top_k: int = 5) -> Dict:
    global _index, _metadata, _ready, _tcode_index_map

    if not query or not query.strip():
        return {"status": "invalid_query", "message": "Query cannot be empty"}

    query = query.strip()

    # 1) Alias exact match FIRST
    alias_exact_matches = _find_alias_exact_match(query)
    if alias_exact_matches:
        results = []
        for m in alias_exact_matches[:top_k]:
            tcode = (m.get("tcode") or "").upper() if m.get("tcode") else None
            meta = _tcode_index_map.get(tcode, {}) if tcode else {}
            desc = meta.get("description") or m.get("description")
            module = meta.get("module") or meta.get("module_name") or m.get("module") or None
            results.append(
                {
                    "tcode": tcode,
                    "description": desc,
                    "module": module,
                    "score": round(float(m.get("score", 1.0)), 6),
                    "match_type": "alias_exact",
                }
            )
        return build_response_with_confidence(results)

    # 2) Alias all-meaningful-words match SECOND
    alias_all_words_matches = _find_alias_all_words_match(query)
    if alias_all_words_matches:
        results = []
        for m in alias_all_words_matches[:top_k]:
            tcode = (m.get("tcode") or "").upper() if m.get("tcode") else None
            meta = _tcode_index_map.get(tcode, {}) if tcode else {}
            desc = meta.get("description") or m.get("description")
            module = meta.get("module") or meta.get("module_name") or m.get("module") or None
            results.append(
                {
                    "tcode": tcode,
                    "description": desc,
                    "module": module,
                    "score": round(float(m.get("score", 0.98)), 6),
                    "match_type": "alias_all_words",
                }
            )
        return build_response_with_confidence(results)

    # 3) Description match THIRD
    description_matches = _find_description_matches(query, top_k=top_k)
    if description_matches:
        return build_response_with_confidence(description_matches)

    # 4) Semantic fallback
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
                    "match_type": "semantic",
                }
            )

        response = build_response_with_confidence(results)

        # Safety guard:
        # if semantic confidence is too low, return suggestions instead of a wrong answer
        if response.get("confidence", 0.0) < 0.5:
            return build_did_you_mean_response(results)

        return response

    except Exception as exc:
        print(f"Search runtime error: {exc}")
        return {"status": "error", "message": f"Search failed: {str(exc)}"}


def search(query: str, top_k: int = 5) -> Dict:
    return search_tcode(query, top_k=top_k)


__all__ = ["initialize_search", "is_ready", "search_tcode", "search"]
