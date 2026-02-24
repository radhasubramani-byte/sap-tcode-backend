import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="SAP T-Code Backend")

# CORS (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import AFTER app creation to avoid circular issues
from app.services.search_service import search_tcode


@app.get("/")
def health_check():
    return {"status": "SAP T-Code backend running"}


@app.get("/search")
def search(query: str):
    results = search_tcode(query)
    return {"results": results}