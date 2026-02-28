# app/main.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import Any, Callable, Optional, Dict
import importlib
import threading
import time
import traceback

app = FastAPI(title="SAP T-Code Assistant API", version="1.0.0")

# -----------------------------
# Dynamic binding to search_service
# -----------------------------
search_module = None
run_search: Optional[Callable[..., Any]] = None
initialize_search_fn: Optional[Callable[[], Any]] = None
is_ready_fn: Optional[Callable[[], bool]] = None
SEARCH_FUNCTION_BOUND = "none"


def _bind_search_functions() -> None:
    """
    Dynamically import app.services.search_service and bind the first available
    search function name to run_search, plus initialize_search/is_ready if present.
    This avoids deploy breaks when function names change.
    """
    global search_module, run_search, initialize_search_fn, is_ready_fn, SEARCH_FUNCTION_BOUND

    try:
        search_module = importlib.import_module("app.services.search_service")
    except Exception:
        search_module = None
        run_search = None
        initialize_search_fn = None
        is_ready_fn = None
        SEARCH_FUNCTION_BOUND = "import_failed"
        print("❌ Failed to import app.services.search_service")
        print(traceback.format_exc())
        return

    # Bind search function (try multiple names)
    candidates = ["search_tcode", "search", "semantic_search", "query"]
    run_search = None
    SEARCH_FUNCTION_BOUND = "none"

    for name in candidates:
        fn = getattr(search_module, name, None)
        if callable(fn):
            run_search = fn
            SEARCH_FUNCTION_BOUND = name
            break

    # Bind init + readiness if available
    initialize_search_fn = getattr(search_module, "initialize_search", None)
    if not callable(initialize_search_fn):
        initialize_search_fn = None

    is_ready_fn = getattr(search_module, "is_ready", None)
    if not callable(is_ready_fn):
        is_ready_fn = None

    print(f"✅ Bound search function: {SEARCH_FUNCTION_BOUND}")
    print(f"✅ initialize_search present: {bool(initialize_search_fn)}")
    print(f"✅ is_ready present: {bool(is_ready_fn)}")


def _background_init() -> None:
    """
    Run initialize_search() in a background thread so Render doesn't time out.
    """
    if not initialize_search_fn:
        print("⚠️ initialize_search() not available — skipping background init.")
        return

    try:
        print("🔄 Background initialization thread starting...")
        initialize_search_fn()
        print("✅ initialize_search() completed")
    except Exception:
        print("❌ initialize_search() crashed")
        print(traceback.format_exc())


# Bind immediately at import time (safe; no heavy loading should happen in search_service)
_bind_search_functions()

# Try to include voice webhook router IF present (optional)
try:
    voice_module = importlib.import_module("app.services.voice_webhook")
    voice_router = getattr(voice_module, "router", None)
    if voice_router is not None:
        app.include_router(voice_router)
        print("✅ voice_webhook router included")
    else:
        print("ℹ️ app.services.voice_webhook found but no router object")
except Exception:
    # Not an error if you haven't created voice_webhook.py yet
    print("ℹ️ voice_webhook not included (file not present or import failed)")

# Startup: kick off background init
@app.on_event("startup")
def on_startup():
    # Re-bind on startup too (helps if hot reload / partial deployments)
    _bind_search_functions()

    t = threading.Thread(target=_background_init, daemon=True)
    t.start()
    print("🧵 Started background initialization thread (daemon)")


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "service": "sap-tcode-backend",
        "status": "ok",
        "endpoints": ["/health", "/search-tcode?q=...&top_k=5"],
    }


@app.get("/health")
def health():
    ready = False
    if is_ready_fn:
        try:
            ready = bool(is_ready_fn())
        except Exception:
            ready = False

    return {
        "ready": ready,
        "search_function_bound": SEARCH_FUNCTION_BOUND,
    }


@app.get("/search-tcode")
def search_tcode(
    q: str = Query(..., min_length=1, description="Natural language query"),
    top_k: int = Query(5, ge=1, le=10, description="Number of results"),
):
    """
    Canonical endpoint for your voice agent + UI.
    Calls the dynamically-bound search function from search_service.py.
    """
    if run_search is None:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Search function is not available (binding failed).",
                "search_function_bound": SEARCH_FUNCTION_BOUND,
            },
        )

    try:
        # Try calling with (query, top_k) first
        try:
            result = run_search(q, top_k=top_k)  # preferred signature
        except TypeError:
            # Fallback for older signature variants
            result = run_search(q, top_k)

        # If your search_service already returns canonical JSON, pass it through.
        # Otherwise, wrap list outputs cleanly.
        if isinstance(result, dict):
            return result
        return {
            "status": "ok",
            "query": q,
            "results": result,
        }
    except Exception as e:
        print("❌ /search-tcode runtime error:", str(e))
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Search failed: {str(e)}",
            },
        )