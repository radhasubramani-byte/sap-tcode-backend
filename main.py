from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Import knowledge loader + search
from app.services.knowledge_loader import load_knowledge
from app.services.search_service import search_tcode

app = FastAPI(title="SAP TCode AI Backend")


# -------------------------------
# STARTUP — LOAD EMBEDDINGS
# -------------------------------
@app.on_event("startup")
def startup_event():
    try:
        print("====================================")
        print("Loading SAP knowledge base...")
        load_knowledge()
        print("SAP Knowledge Loaded Successfully")
        print("====================================")
    except Exception as e:
        print("FAILED TO LOAD KNOWLEDGE BASE")
        traceback.print_exc()


# -------------------------------
# REQUEST MODEL
# -------------------------------
class SearchRequest(BaseModel):
    query: str


# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/")
def root():
    return {"status": "SAP TCode AI running"}


# -------------------------------
# SEARCH ENDPOINT (FOR VAPI)
# -------------------------------
@app.post("/search-tcode")
def search_endpoint(payload: SearchRequest):
    try:
        if not payload.query or len(payload.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query required")

        result = search_tcode(payload.query)

        if not result:
            return {
                "found": False
            }

        return {
            "found": True,
            "tcode": result["tcode"],
            "description": result["description"],
            "module": result.get("module", ""),
            "confidence": str(result.get("score", 0.9))
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Search failed")