# app/main.py
from __future__ import annotations

import os
import threading
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Import voice webhook router (this is the one we updated) ---
try:
    from app.voice_webhook import router as voice_router
except Exception as e:
    voice_router = None
    print("❌ Failed to import app.voice_webhook router:", repr(e))
    traceback.print_exc()

# --- Import search service (semantic search init) ---
try:
    from app.services import search_service
except Exception as e:
    search_service = None
    print("⚠️ Failed to import search_service:", repr(e))

# --- (Optional) Import your other API router if you have one ---
# If you don't have app/api.py, this won't break startup.
try:
    from app.api import router as api_router  # optional
except Exception:
    api_router = None

app = FastAPI(title="SAP T-Code Backend", version="1.0.0")

# CORS (safe default; tighten later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Routes
# -----------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "sap-tcode-backend"}

@app.get("/health")
def health():
    ready = False
    try:
        if search_service and hasattr(search_service, "is_ready"):
            ready = bool(search_service.is_ready())
    except Exception:
        ready = False
    return {"status": "ok", "search_ready": ready}

# Mount voice webhook
if voice_router is not None:
    app.include_router(voice_router)
    print("✅ Mounted voice webhook routes (e.g., POST /voice-webhook)")
else:
    print("❌ voice_router is None — /voice-webhook will NOT be available")

# Mount optional API router (if exists)
if api_router is not None:
    app.include_router(api_router)
    print("✅ Mounted API routes from app.api")
else:
    print("ℹ️ app.api router not found (skipping)")

# -----------------------
# Startup: initialize search in background
# -----------------------
def _init_search_background():
    if not search_service:
        print("⚠️ search_service unavailable — skipping initialize_search()")
        return
    if not hasattr(search_service, "initialize_search"):
        print("⚠️ search_service.initialize_search not found — skipping")
        return

    try:
        search_service.initialize_search()
        print("✅ initialize_search() completed")
    except Exception as e:
        print("❌ initialize_search() failed:", repr(e))
        traceback.print_exc()

@app.on_event("startup")
def on_startup():
    # Helpful banner for Render logs
    primary_url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("PRIMARY_URL") or ""
    if primary_url:
        print("==> " + "/" * 60)
        print(f"==> Available at your primary URL {primary_url}")
        print("==> " + "/" * 60)

    # Render expects you to listen on PORT
    port = os.environ.get("PORT", "10000")
    print(f"==> Detected service running on port {port}")

    # Kick off background init so first request doesn't block too long
    t = threading.Thread(target=_init_search_background, daemon=True)
    t.start()

    print("==> Your service is live 🎉")


# NOTE:
# You do NOT run uvicorn here in Render; Render runs it via your start command.
# Typical Render start command:
#   uvicorn app.main:app --host 0.0.0.0 --port $PORT