from fastapi import FastAPI
from app.services.search_service import search_tcode
import threading
import time

app = FastAPI(title="SAP TCode Assistant")


# -----------------------------
# Load search engine background
# -----------------------------
def initialize_search():
    time.sleep(2)
    print("Semantic search initialized")


@app.on_event("startup")
def startup_event():
    print("Starting background initialization...")
    thread = threading.Thread(target=initialize_search, daemon=True)
    thread.start()


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {
        "ready": True,
        "search_function_bound": "search_tcode"
    }


# -----------------------------
# Search Endpoint (existing)
# -----------------------------
@app.get("/search-tcode")
def search(q: str, top_k: int = 5):
    return search_tcode(q, top_k)


# -----------------------------
# Voice Webhook (NEW)
# -----------------------------
try:
    from app.voice_webhook import router as voice_router
    app.include_router(voice_router)
    print("voice_webhook router loaded successfully")
except Exception as e:
    print("FAILED to load voice_webhook:", e)