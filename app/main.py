# app/main.py
"""
FastAPI entrypoint for SAP T-code Virtual Consultant backend.

This version uses defensive imports so main.py remains compatible across
multiple versions of app.services.search_service (helps when reverting).
It:
- Starts initialize_search() in a background daemon thread on startup (non-blocking)
- Exposes /health and /search endpoints
- Adds debug endpoints:
    - GET  /__debug/search_status       -> returns is_ready + index meta (graceful fallback)
    - POST /__debug/initialize_search   -> triggers initialization manually
- Uses run_in_threadpool for CPU-bound search calls
"""

from typing import Any, Dict
import threading
import traceback
import importlib

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

app = FastAPI(title="SAP T-Code Virtual Consultant Backend (defensive main)")

# Allow CORS for testing / demo. Restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#
# Defensive import of the search service module and its callable members.
# We import the module as a whole and then extract attributes with getattr to
# avoid ImportError crashes when a particular helper is not present.
#
_search_module = importlib.import_module("app.services.search_service")

# required API entries (attempt to resolve; provide fallbacks where reasonable)
initialize_search = getattr(_search_module, "initialize_search", None)
is_ready = getattr(_search_module, "is_ready", lambda: False)
# prefer search_tcode, then search; eventually other callers may use search()
search_tcode = getattr(_search_module, "search_tcode", getattr(_search_module, "search", None))
# optional debug accessor; fallback returns a simple "Not Found" dict
_get_index_meta = getattr(_search_module, "_get_index_meta", lambda: {"detail": "Not Found"})

# sanity checks: ensure required callables exist
if initialize_search is None:
    # initialize_search is optional but recommended; warn in logs
    print("WARNING: initialize_search not found in search_service; background init disabled.")
if search_tcode is None:
    raise ImportError("search_service does not expose 'search_tcode' or 'search' - cannot continue")


@app.on_event("startup")
def _start_background_tasks():
    """
    Start the search initialization in a daemon thread so that the process
    becomes healthy immediately while the heavy work runs in background.
    """
    try:
        if callable(initialize_search):
            t = threading.Thread(target=initialize_search, daemon=True)
            t.start()
            print("Started background thread to initialize semantic search.")
        else:
            print("initialize_search not callable; skipping background initialization.")
    except Exception as e:
        print("Failed to start initialize_search background thread:", str(e))
        traceback.print_exc()


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Basic health endpoint. Returns HTTP 200 if the app is running.
    Include search readiness as `ready` (True when embeddings/index are loaded).
    """
    return {"status": "healthy", "ready": bool(is_ready())}


@app.get("/search")
async def search(q: str = Query(None, alias="q"), top_k: int = Query(5, alias="top_k")):
    """
    Search endpoint. Example:
      GET /search?q=create%20purchase%20requisition&top_k=3

    Returns:
      - warming_up status if the search index is not ready
      - invalid_query if q is empty
      - results (list or typed dict) from the search service on success
      - error object on failure
    """
    if q is None or not q.strip():
        return JSONResponse(
            status_code=400,
            content={"status": "invalid_query", "message": "Query parameter 'q' is required."},
        )

    try:
        # run the (potentially) CPU-bound search in a threadpool to avoid blocking the event loop
        result = await run_in_threadpool(search_tcode, q, int(top_k))
        # If the underlying search returns a list/dict, forward as-is
        return JSONResponse(status_code=200, content=result)
    except Exception as exc:
        tb = traceback.format_exc()
        print("Search endpoint runtime error:", str(exc))
        print(tb)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Search failed", "detail": str(exc)},
        )


# -------------------------
# Debug / diagnostic helpers
# -------------------------
@app.get("/__debug/search_status")
def debug_search_status():
    """
    Returns basic readiness and index metadata for debugging.
    Gracefully handles older/newer versions of the search service.
    """
    try:
        ready = bool(is_ready())
        try:
            info = _get_index_meta() if callable(_get_index_meta) else _get_index_meta
        except Exception:
            info = {"detail": "Not Found"}
        return {"ready": ready, "index_info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failure: {str(e)}")


@app.post("/__debug/initialize_search")
def debug_initialize_search():
    """
    Manually trigger initialization (idempotent). Useful when startup thread didn't run.
    """
    try:
        if callable(is_ready) and is_ready():
            return {"status": "already_ready"}
        if not callable(initialize_search):
            raise HTTPException(status_code=500, detail="initialize_search not available in this deployment")
        threading.Thread(target=initialize_search, daemon=True).start()
        return {"status": "initialization_started"}
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("Failed to start initialization:", str(e))
        print(tb)
        raise HTTPException(status_code=500, detail=f"Failed to start initialization: {str(e)}")