# app/main.py
import threading
from typing import Any, Callable
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# safe imports that should exist in the defensive search_service
from app.services.search_service import initialize_search, is_ready

# Try multiple possible names for the search function so main.py works with either version.
search_fn: Callable[[str, int], Any] | None = None
try:
    # preferred name used in older lightweight versions
    from app.services.search_service import search as _search
    search_fn = _search
except Exception:
    try:
        from app.services.search_service import search_tcode as _search
        search_fn = _search
    except Exception:
        try:
            from app.services.search_service import semantic_search as _search
            search_fn = _search
        except Exception:
            # leave search_fn = None; endpoint will handle missing function gracefully
            search_fn = None


app = FastAPI(title="SAP T-Code Semantic Search API (robust main)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def background_loader():
    print("🔄 Initializing semantic search engine (background thread)...")
    try:
        initialize_search()
        print("✅ Background initialization complete")
    except Exception as e:
        print("❌ Background initialization failed:", e)


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=background_loader, daemon=True)
    thread.start()
    print("Started background thread to initialize semantic search.")


@app.get("/")
def root():
    return {
        "service": "SAP TCode Assistant",
        "status": "running",
        "semantic_search_ready": is_ready(),
        "search_function_available": bool(search_fn)
    }


@app.get("/health")
def health():
    return {
        "ready": is_ready(),
        "search_function_available": bool(search_fn),
        "message": "embeddings_loaded" if is_ready() else "loading"
    }


@app.get("/search-tcode")
def search_tcode(q: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=20)):
    """
    Main search endpoint.
    - Calls whichever search function is available in search_service.
    - Returns warmup/diagnostic payloads if the system isn't ready or search function missing.
    Example:
      /search-tcode?q=create+purchase+requisition&top_k=5
    """
    # sanity
    if not search_fn:
        return {
            "status": "error",
            "message": "No search function available in app.services.search_service. Expected one of: search, search_tcode, semantic_search"
        }

    if not is_ready():
        return {
            "status": "warming_up",
            "message": "AI knowledge base loading — try again in a few seconds"
        }

    try:
        # call signature may be search(query) or search(query, top_k)
        try:
            results = search_fn(q, top_k)
        except TypeError:
            results = search_fn(q)

        return {
            "status": "ok",
            "query": q,
            "results": results
        }
    except Exception as exc:
        # don't crash on a runtime error; return structured message for debugging
        return {
            "status": "error",
            "message": str(exc)
        }