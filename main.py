from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.search_service import search_tcode
import os

app = FastAPI(title="SAP TCode Backend")

# Allow all origins (for Vercel frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Root endpoint (Render check)
# -----------------------------
@app.get("/")
def root():
    return {
        "service": "SAP TCode Search API",
        "status": "running"
    }

# -----------------------------
# Health check (IMPORTANT)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Search endpoint
# -----------------------------
@app.get("/search")
def search(query: str):
    results = search_tcode(query)
    return {
        "query": query,
        "count": len(results),
        "results": results
    }