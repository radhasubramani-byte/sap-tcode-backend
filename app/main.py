from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading

# Import search service safely (supports different versions)
try:
    from app.services.search_service import search as run_search
except:
    try:
        from app.services.search_service import semantic_search as run_search
    except:
        from app.services.search_service import query as run_search

from app.services.search_service import initialize_search, is_ready

app = FastAPI(title="SAP T-Code Assistant API")


# -------------------------------------------------------
# Allow VAPI / Browser / Postman access
# -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# Request Model
# -------------------------------------------------------
class SearchRequest(BaseModel):
    query: str


# -------------------------------------------------------
# Background startup (loads embeddings)
# -------------------------------------------------------
@app.on_event("startup")
def startup_event():
    print("🚀 Starting SAP semantic engine in background...")

    def background_load():
        try:
            initialize_search()
            print("✅ Semantic search ready")
        except Exception as e:
            print("❌ Failed to initialize search:", e)

    thread = threading.Thread(target=background_load, daemon=True)
    thread.start()


# -------------------------------------------------------
# Root endpoint (for browser test)
# -------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "SAP T-Code Assistant",
        "status": "running",
        "ready": is_ready()
    }


# -------------------------------------------------------
# Health endpoint (Render health check)
# -------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "semantic_ready": is_ready()
    }


# -------------------------------------------------------
# Main search endpoint (USED BY VOICE AGENT)
# -------------------------------------------------------
@app.post("/search-tcode")
def search_tcode(req: SearchRequest):

    if not is_ready():
        return {
            "success": False,
            "message": "Semantic engine still loading. Try again in 10 seconds."
        }

    try:
        results = run_search(req.query)

        if not results:
            return {
                "success": True,
                "confidence": 0,
                "results": [],
                "message": "No SAP transaction found"
            }

        return {
            "success": True,
            "results": results
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }