# app/main.py
import importlib
import threading
from typing import Any, Callable, Optional, List
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Load module (import module object only; don't import symbols that might not exist) ---
svc_mod = importlib.import_module("app.services.search_service")

# --- Helper accessors with graceful fallbacks ---
initialize_search: Optional[Callable[[], None]] = getattr(svc_mod, "initialize_search", None)
is_ready: Callable[[], bool] = getattr(svc_mod, "is_ready", lambda: False)

# Try several candidate function names for the semantic search function.
_search_candidates: List[str] = ["search", "search_tcode", "semantic_search", "query", "_search", "search_query"]
search_fn: Optional[Callable[..., Any]] = None
for name in _search_candidates:
    candidate = getattr(svc_mod, name, None)
    if callable(candidate):
        search_fn = candidate
        _bound_name = name
        break
else:
    _bound_name = None

app = FastAPI(title="SAP T-Code Assistant (robust startup)")

# CORS for UI / VAPI / testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for POST
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# Background initializer
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

# Root for quick browser test
@app.get("/")
def root():
    return {
        "service": "SAP T-Code Assistant",
        "status": "running",
        "semantic_ready": is_ready() if callable(is_ready) else False,
        "search_function_bound": _bound_name
    }

# Health endpoint used by Render / monitoring
@app.get("/health")
def health():
    return {
        "ready": is_ready() if callable(is_ready) else False,
        "search_function_bound": _bound_name
    }

# Generic helper to call the discovered search function safely
def _call_search(q: str, top_k: int = 5):
    if not search_fn:
        raise RuntimeError("No search function bound in search_service module. Expected one of: " + ", ".join(_search_candidates))

    # Try calling with (query, top_k) first, then fallback to (query)
    try:
        return search_fn(q, top_k)
    except TypeError:
        return search_fn(q)

# GET-based convenience endpoint
@app.get("/search-tcode")
def search_get(q: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=50)):
    if not callable(is_ready) or not is_ready():
        return {"status": "warming_up", "message": "Semantic engine initializing"}
    try:
        results = _call_search(q, top_k)
        return {"status": "ok", "query": q, "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# POST-based endpoint for voice agent / VAPI
@app.post("/search-tcode")
def search_post(body: SearchRequest = Body(...)):
    q = body.query
    top_k = body.top_k or 5
    if not callable(is_ready) or not is_ready():
        return {"status": "warming_up", "message": "Semantic engine initializing"}
    try:
        results = _call_search(q, top_k)
        return {"status": "ok", "query": q, "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}