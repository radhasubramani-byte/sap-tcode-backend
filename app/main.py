# app/main.py
import os
import logging
import threading
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="SAP T-code Assistant Backend")

# attempt to import the initialize_search + whichever search function is available
initialize_search = None
search_fn = None
bound_search_name = None

try:
    from app.services import search_service  # type: ignore
    initialize_search = getattr(search_service, "initialize_search", None)
    # pick a search function name from a set of common names
    for nm in ("search_tcode", "search", "semantic_search", "query"):
        maybe = getattr(search_service, nm, None)
        if callable(maybe):
            search_fn = maybe
            bound_search_name = nm
            break
    logger.info("main: bound search function: %s", bound_search_name)
except Exception as e:
    logger.warning("main: could not import search_service at startup: %s", e)
    initialize_search = None
    search_fn = None

# include voice_webhook router
try:
    from app.voice_webhook import router as voice_router  # type: ignore
    app.include_router(voice_router)
    logger.info("main: voice_webhook router included")
except Exception as e:
    logger.warning("main: could not include voice_webhook router: %s", e)


@app.on_event("startup")
def startup_event():
    """
    Start background initialization of search so the app can become healthy quickly while search warms up.
    """
    if callable(initialize_search):
        def _init():
            try:
                logger.info("main: starting initialize_search() in background thread")
                initialize_search()
                logger.info("main: initialize_search() completed")
            except Exception as ex:
                logger.exception("main: initialize_search() failed: %s", ex)

        t = threading.Thread(target=_init, daemon=True)
        t.start()
    else:
        logger.warning("main: initialize_search function not available at startup")


@app.get("/")
async def root():
    return JSONResponse({"detail": "Not Found"}, status_code=404)


@app.get("/health")
async def health():
    """
    Health endpoint used by render / monitors.
    Return readiness and which search function name is bound (if any).
    """
    ready = False
    try:
        # if the search_service exposes is_ready(), use it
        from app.services import search_service  # type: ignore
        is_ready = getattr(search_service, "is_ready", None)
        if callable(is_ready):
            ready = bool(is_ready())
    except Exception:
        ready = False

    return {"ready": ready, "search_function_bound": bound_search_name or None}


@app.post("/search-tcode")
async def search_tcode(request: Request):
    """
    A simple POST wrapper that forwards to the bound internal search function (if any).
    Accepts either:
      - JSON body with {"q": "query string", "top_k": 5}
    or
      - query param ?q=...
    Returns whatever the internal search returns, with a defensive wrapper on errors.
    """
    body: Dict[str, Any] = {}
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
    except Exception:
        body = {}

    q = body.get("q") or request.query_params.get("q") or ""
    top_k = int(body.get("top_k") or request.query_params.get("top_k") or 5)

    if not q:
        raise HTTPException(status_code=422, detail="Field required: q")

    # call bound search function if present
    if callable(search_fn):
        try:
            result = search_fn(q, top_k=top_k) if search_fn.__code__.co_argcount >= 1 else search_fn(q)
            # support coroutine result
            if hasattr(result, "__await__"):
                result = await result
            # if function returned a dict, return it
            if isinstance(result, dict):
                return result
            # if list, wrap
            if isinstance(result, (list, tuple)):
                return {"status": "ok", "query": q, "results": result, "best_match": (result[0] if result else None)}
            return {"status": "ok", "query": q, "results": result, "best_match": None}
        except Exception as ex:
            logger.exception("main: internal search_fn failed: %s", ex)
            raise HTTPException(status_code=500, detail="Search engine error")
    else:
        # If no internal function bound, attempt to call a local endpoint (defensive)
        import httpx
        fallback_url = os.getenv("LOCAL_SEARCH_FALLBACK_URL", "http://127.0.0.1:10000/search-tcode")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(fallback_url, params={"q": q, "top_k": top_k})
                r.raise_for_status()
                return r.json()
        except Exception as e:
            logger.exception("main: fallback search HTTP call failed: %s", e)
            raise HTTPException(status_code=500, detail="Search engine not available")


# allow simple run with: uvicorn app.main:app --host 0.0.0.0 --port 10000
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")