# app/main.py
"""
Compatibility main that imports the search_service module (safe)
and binds any available search function name. Normalizes outputs
so callers get a single canonical JSON shape.
"""

import importlib
import threading
from typing import Any, Callable, Dict, Optional
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the module (do not import specific names that might not exist)
svc_mod = importlib.import_module("app.services.search_service")

# optional functions
initialize_search = getattr(svc_mod, "initialize_search", None)
is_ready = getattr(svc_mod, "is_ready", lambda: False)

# Try to bind a search function from a list of common names
_search_candidates = ["search", "search_tcode", "semantic_search", "query", "_search", "search_query"]
search_fn = None
_bound_name = None
for name in _search_candidates:
    candidate = getattr(svc_mod, name, None)
    if callable(candidate):
        search_fn = candidate
        _bound_name = name
        break

app = FastAPI(title="SAP T-Code Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

def _background_initialize():
    print("🔄 Background initialization thread starting...")
    try:
        if callable(initialize_search):
            initialize_search()
            print("✅ initialize_search() completed")
        else:
            print("⚠️ initialize_search() not found in search_service module")
    except Exception as e:
        print("❌ initialize_search() raised:", repr(e))

@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=_background_initialize, daemon=True)
    t.start()
    print("Started background initialization thread (daemon)")

@app.get("/")
def root():
    return {
        "service": "SAP T-Code Assistant",
        "status": "running",
        "semantic_ready": is_ready() if callable(is_ready) else False,
        "search_function_bound": _bound_name
    }

@app.get("/health")
def health():
    return {
        "ready": is_ready() if callable(is_ready) else False,
        "search_function_bound": _bound_name
    }

def _normalize_search_output(raw: Any) -> Dict:
    """
    Normalize raw search output (either list or dict) to canonical dict:
    {
      "status": "ok"|"warming_up"|"error",
      "query": "...",
      "results": [...],
      "best_match": {...} or None,
      "confidence": float or None,
      "confidence_label": str or None
    }
    """
    canonical = {
        "status": "ok",
        "query": None,
        "results": [],
        "best_match": None,
        "confidence": None,
        "confidence_label": None
    }

    if raw is None:
        return canonical

    # If service already returned a status-style dict, preserve it
    if isinstance(raw, dict):
        # If it contains a status like "warming_up" or "error", pass through
        if raw.get("status") in ("warming_up", "invalid_query", "error"):
            return raw
        # If raw is rich response from build_response_with_confidence
        if "results" in raw and isinstance(raw["results"], list):
            canonical["results"] = raw["results"]
            canonical["best_match"] = raw.get("best_match")
            canonical["confidence"] = raw.get("confidence")
            canonical["confidence_label"] = raw.get("confidence_label")
            return canonical
        # else if it's already a list placed inside another key - try to coerce
        if "alternatives" in raw and isinstance(raw["alternatives"], list):
            canonical["results"] = raw["alternatives"]
            canonical["best_match"] = raw.get("best_match")
            canonical["confidence"] = raw.get("confidence")
            canonical["confidence_label"] = raw.get("confidence_label")
            return canonical

    # if raw is list-like (old format)
    if isinstance(raw, (list, tuple)):
        canonical["results"] = list(raw)
        if len(canonical["results"]) > 0:
            top = canonical["results"][0]
            canonical["best_match"] = top
            score = (top.get("score") if isinstance(top, dict) else None)
            try:
                sc = float(score) if score is not None else None
                if sc is not None:
                    canonical["confidence"] = round(sc, 3)
                    if sc >= 0.70:
                        canonical["confidence_label"] = "high"
                    elif sc >= 0.40:
                        canonical["confidence_label"] = "medium"
                    else:
                        canonical["confidence_label"] = "low"
            except Exception:
                pass
    return canonical

def _call_search(q: str, top_k: int = 5):
    if not search_fn:
        raise RuntimeError("No search function bound in search_service module. Expected one of: " + ", ".join(_search_candidates))
    # try (query, top_k) signature, then (query) fallback
    try:
        return search_fn(q, top_k)
    except TypeError:
        return search_fn(q)

@app.get("/search-tcode")
def search_get(q: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=50)):
    if not callable(is_ready) or not is_ready():
        return {"status": "warming_up", "message": "Semantic engine initializing"}
    try:
        raw = _call_search(q, top_k)
        canonical = _normalize_search_output(raw)
        canonical["query"] = q
        return canonical
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/search-tcode")
def search_post(body: SearchRequest = Body(...)):
    return search_get(q=body.query, top_k=body.top_k or 5)