from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import os

# IMPORTANT: absolute imports from app package
from app.services.search_service import search_tcode
from app.services.knowledge_loader import load_knowledge

app = FastAPI(title="SAP TCode Search API")


# ---------- STARTUP ----------
@app.on_event("startup")
def startup_event():
    print("🚀 Starting SAP TCode API...")
    load_knowledge()
    print("✅ Knowledge loaded successfully")


# ---------- HEALTH CHECK ----------
@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------- ROOT ----------
@app.get("/")
def root():
    return {"message": "SAP TCode API running"}


# ---------- SEARCH ----------
@app.get("/search")
def search(q: str = Query(..., description="Search query")):
    try:
        results = search_tcode(q)

        return JSONResponse(
            content={
                "query": q,
                "count": len(results),
                "results": results
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )