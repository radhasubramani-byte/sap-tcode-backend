from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import threading
import time

# Import ONLY safe functions
from app.services.search_service import (
    initialize_search,
    is_ready,
    search
)

app = FastAPI(title="SAP T-Code Semantic Search API")


# =========================================================
# CORS (Voice agent + frontend safe)
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Background Loader (Prevents Render timeout)
# =========================================================
def background_loader():
    print("🔄 Initializing semantic search engine...")
    try:
        initialize_search()
        print("✅ Semantic search ready")
    except Exception as e:
        print("❌ Failed to initialize search:", e)


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=background_loader, daemon=True)
    thread.start()
    print("Started background thread to initialize semantic search.")


# =========================================================
# Root (Render health ping)
# =========================================================
@app.get("/")
def root():
    return {"service": "SAP TCode Assistant", "status": "running"}


# =========================================================
# Health Endpoint (IMPORTANT FOR VAPI + UI)
# =========================================================
@app.get("/health")
def health():
    return {
        "ready": is_ready(),
        "message": "Embeddings loaded" if is_ready() else "Loading embeddings"
    }


# =========================================================
# Main Search Endpoint
# =========================================================
@app.get("/search-tcode")
def search_tcode(q: str = Query(..., min_length=2)):
    if not is_ready():
        return {
            "status": "loading",
            "message": "Knowledge base is still initializing"
        }

    results = search(q, top_k=5)

    return {
        "status": "ok",
        "query": q,
        "results": results
    }