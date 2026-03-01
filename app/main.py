# app/main.py
from __future__ import annotations

import importlib
import threading
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SAP T-code Backend", version="1.0.0")

# (Optional) helps browser-based tool testers; safe to keep permissive during dev.
# If you want stricter later, restrict allow_origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Bind search_service safely
# -------------------------
SEARCH_FN: Optional[Callable[..., Any]] = None
INIT_FN: Optional[Callable[[], Any]] = None

def _bind_search_functions() -> None:
    global SEARCH_FN, INIT_FN
    m = importlib.import_module("app.services.search_service")

    # init function
    INIT_FN = getattr(m, "initialize_search", None)

    # search function names across versions
    SEARCH_FN = (
        getattr(m, "search_tcode", None)
        or getattr(m, "search", None)
        or getattr(m, "semantic_search", None)
        or getattr(m, "query", None)
    )

def _start_background_init() -> None:
    if callable(INIT_FN):
        def runner():
            try:
                INIT_FN()  # type: ignore[misc]
                print("✅ initialize_search() completed")
            except Exception as e:
                print(f"❌ initialize_search() failed: {e}")

        t = threading.Thread(target=runner, daemon=True)
        t.start()

# -------------------------
# Include voice webhook router
# -------------------------
try:
    from app.voice_webhook import router as voice_router
    app.include_router(voice_router)
except Exception as e:
    print(f"WARNING: main: could not include voice_webhook router: {e}")

@app.on_event("startup")
def on_startup() -> None:
    _bind_search_functions()
    _start_background_init()

@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "sap-tcode-backend", "endpoints": ["/health", "/search-tcode", "/voice-webhook"]}

@app.get("/health")
def health() -> Dict[str, Any]:
    # Ask search_service if ready if available
    try:
        m = importlib.import_module("app.services.search_service")
        is_ready = getattr(m, "is_ready", None)
        ready = bool(is_ready()) if callable(is_ready) else False
    except Exception:
        ready = False

    bound_name = None
    if SEARCH_FN is not None:
        bound_name = getattr(SEARCH_FN, "__name__", "unknown")

    return {"ready": ready, "search_function_bound": bound_name}

@app.get("/search-tcode")
def search_tcode(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=10),
) -> Dict[str, Any]:
    if not callable(SEARCH_FN):
        return {"status": "error", "message": "Search function not available"}

    try:
        payload = SEARCH_FN(q, top_k=top_k)  # type: ignore[misc]
        # If your search_service returns list, wrap it
        if isinstance(payload, list):
            return {"status": "ok", "query": q, "results": payload}
        return payload
    except Exception as e:
        return {"status": "error", "message": str(e)}