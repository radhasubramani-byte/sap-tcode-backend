import threading
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Import ONLY safe functions (no heavy imports at startup)
from app.services.search_service import (
    initialize_search,
    is_ready,
    semantic_search
)

# =========================================================
# FastAPI App
# =========================================================
app = FastAPI(title="SAP T-Code AI Assistant", version="1.0")


# Allow frontend / VAPI / browser calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Background Initialization (NON-BLOCKING STARTUP)
# =========================================================
def start_background_initialization():
    print("🔄 Initializing semantic search engine... (defensive)")
    try:
        initialize_search()
        print("✅ Semantic search initialization finished")
    except Exception as e:
        print(f"❌ Failed to initialize search: {e}")


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=start_background_initialization, daemon=True)
    thread.start()
    print("Started background thread to initialize semantic search.")


# =========================================================
# Root Endpoint (for Render + Demo)
# =========================================================
@app.get("/")
def root():
    return {
        "service": "SAP T-Code AI Assistant",
        "status": "running",
        "semantic_search_ready": is_ready()
    }


# =========================================================
# Health Check Endpoint
# =========================================================
@app.get("/health")
def health():
    return {
        "healthy": True,
        "embeddings_loaded": is_ready()
    }


# =========================================================
# T-Code Semantic Search Endpoint
# =========================================================
@app.get("/search-tcode")
def search_tcode(q: str = Query(..., description="Describe SAP task")):
    """
    Example:
    /search-tcode?q=create purchase order
    """
    if not is_ready():
        return {
            "ready": False,
            "message": "Semantic search still initializing. Try again in a few seconds."
        }

    try:
        results = semantic_search(q)

        return {
            "ready": True,
            "query": q,
            "results": results
        }

    except Exception as e:
        return {
            "ready": False,
            "error": str(e)
        }


# =========================================================
# Debug Retrieval Endpoint (for presentation demo)
# Shows raw retrieval before LLM answer
# =========================================================
@app.get("/debug-search")
def debug_search(q: str):
    if not is_ready():
        return {"status": "initializing"}

    results = semantic_search(q)
    return {
        "query": q,
        "retrieved_knowledge": results
    }