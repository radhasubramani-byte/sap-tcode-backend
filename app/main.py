# app/main.py
"""
FastAPI entrypoint for SAP T-code Virtual Consultant backend.

Features in this updated file:
- Starts initialize_search() in a background daemon thread on startup (non-blocking)
- /health endpoint exposes both service health and search readiness
- /search endpoint (GET) accepts q (query) and top_k and runs search in threadpool
- Debug endpoints:
    - GET  /__debug/search_status       -> returns is_ready + index meta
    - POST /__debug/initialize_search   -> triggers initialization manually
- Defensive error handling and clear JSON responses

Note: remove the debug endpoints once debugging is complete.
"""

from typing import Any, Dict, List
import threading
import traceback

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

# Import the search service functions (search_tcode alias exists in search_service)
from app.services.search_service import (
    initialize_search,
    is_ready,
    _get_index_meta,
    search_tcode,
)

app = FastAPI(title="SAP T-Code Virtual Consultant Backend")

# Optional: allow CORS for local testing / dashboards (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _start_background_tasks():
    """
    Start the search initialization in a daemon thread so that the process
    becomes healthy immediately while the heavy work runs in background.
    """
    try:
        t = threading.Thread(target=initialize_search, daemon=True)
        t.start()
        print("Started background thread to initialize semantic search.")
    except Exception as e:
        print("Failed to start initialize_search background thread:", str(e))
        traceback.print_exc()


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Basic health endpoint. Returns HTTP 200 if the app is running.
    Include search readiness as `ready` (True when embeddings/index are loaded).
    """
    return {"status": "healthy", "ready": is_ready()}


@app.get("/search")
async def search(q: str = Query(None, alias="q"), top_k: int = Query(5, alias="top_k")):
    """
    Search endpoint. Example:
      GET /search?q=create%20purchase%20requisition&top_k=3

    Returns:
      - warming_up status if the search index is not ready
      - invalid_query if q is empty
      - list of {tcode, description, score} on success
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
        return JSONResponse(status_code=200, content=result)
    except Exception as exc:
        # Catch-all to keep service alive and provide debugging info
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
    Remove in production.
    """
    try:
        info = _get_index_meta()
        return {"ready": is_ready(), "index_info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failure: {str(e)}")


@app.post("/__debug/initialize_search")
def debug_initialize_search():
    """
    Manually trigger initialization (idempotent). Useful when startup thread didn't run.
    """
    try:
        if is_ready():
            return {"status": "already_ready"}
        threading.Thread(target=initialize_search, daemon=True).start()
        return {"status": "initialization_started"}
    except Exception as e:
        tb = traceback.format_exc()
        print("Failed to start initialization:", str(e))
        print(tb)
        raise HTTPException(status_code=500, detail=f"Failed to start initialization: {str(e)}")