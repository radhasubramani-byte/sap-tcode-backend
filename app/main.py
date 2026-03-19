# app/main.py
import os
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse

# Search service (semantic)
from app.services.search_service import initialize_search, is_ready, search_tcode

app = FastAPI(title="SAP Tcode Backend", version="1.0.0")


# ----------------------------
# Startup: warm up search index
# ----------------------------
@app.on_event("startup")
def _startup() -> None:
    # Do NOT block startup; run embeddings/index build in background.
    def _bg_init():
        try:
            initialize_search()
        except Exception as e:
            print("❌ initialize_search() failed:", repr(e))

    t = threading.Thread(target=_bg_init, daemon=True)
    t.start()


# ----------------------------
# Basic routes
# ----------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "sap-tcode-backend",
        "ready": bool(is_ready()),
        "interfaces": {
            "voice": "enabled",
            "chat": "enabled",
            "chat_ui": "enabled",
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ready": bool(is_ready())}


# ----------------------------
# Chat UI route
# ----------------------------
@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui():
    try:
        file_path = os.path.join(os.path.dirname(__file__), "static", "chat.html")
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(
            content=f"<h3>Failed to load chat UI: {str(e)}</h3>",
            status_code=500,
        )


# ----------------------------
# SEARCH ENDPOINT
# Existing shared backend search endpoint
# ----------------------------
@app.get("/search-tcode")
def search_tcode_endpoint(
    q: str = Query(..., description="User query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
) -> JSONResponse:
    try:
        resp = search_tcode(q, top_k=top_k)
        return JSONResponse(content=resp)
    except Exception as e:
        print("❌ /search-tcode error:", repr(e))
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "search failed"},
        )


# ----------------------------
# Mount voice webhook routes (defensive)
# Supports either:
#   - voice_webhook.py defines `router` (FastAPI APIRouter)
#   - voice_webhook.py defines `mount_*` or `register_*` function
# ----------------------------
def _mount_voice_routes() -> None:
    try:
        from app.voice_webhook import router as voice_router  # type: ignore

        app.include_router(voice_router)
        print("✅ Mounted voice webhook routes (router)")
        return
    except Exception as e:
        print("ℹ️ voice router import failed:", repr(e))

    for fn_name in ("mount_voice_routes", "register_voice_routes", "register_voice_webhook_routes"):
        try:
            mod = __import__("app.voice_webhook", fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                fn(app)
                print(f"✅ Mounted voice webhook routes ({fn_name})")
                return
        except Exception:
            continue

    print("ℹ️ voice webhook module found but no router/mount function detected (skipping)")


# ----------------------------
# Mount chat webhook routes (defensive)
# Supports either:
#   - chat_webhook.py defines `router` (FastAPI APIRouter)
#   - chat_webhook.py defines `mount_*` or `register_*` function
# ----------------------------
def _mount_chat_routes() -> None:
    try:
        from app.chat_webhook import router as chat_router  # type: ignore

        app.include_router(chat_router)
        print("✅ Mounted chat webhook routes (router)")
        return
    except Exception as e:
        print("ℹ️ chat router import failed:", repr(e))

    for fn_name in ("mount_chat_routes", "register_chat_routes", "register_chat_webhook_routes"):
        try:
            mod = __import__("app.chat_webhook", fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                fn(app)
                print(f"✅ Mounted chat webhook routes ({fn_name})")
                return
        except Exception:
            continue

    print("ℹ️ chat webhook module found but no router/mount function detected (skipping)")


_mount_voice_routes()
_mount_chat_routes()