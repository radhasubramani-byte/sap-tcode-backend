from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import csv
import os
import re

app = FastAPI(title="SAP TCode Search API")

# Allow VAPI / browser calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Data Model
# ----------------------------
class SearchRequest(BaseModel):
    query: str

# ----------------------------
# Load CSV (comma separated)
# ----------------------------
TCODES = []
CSV_PATH = "tcodes.csv"

if not os.path.exists(CSV_PATH):
    print("WARNING: tcodes.csv not found in root directory")
else:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tcode = (row.get("tcode") or "").strip()
            desc = (row.get("description") or "").strip()
            module = (row.get("module") or "").strip()

            if tcode and desc:
                TCODES.append({
                    "tcode": tcode.upper(),
                    "description": desc,
                    "module": module
                })

print(f"Loaded {len(TCODES)} tcodes")

# ----------------------------
# Text Helpers
# ----------------------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def score_match(query: str, description: str) -> float:
    q = normalize(query)
    d = normalize(description)

    if q in d:
        return 1.0

    q_words = set(q.split())
    d_words = set(d.split())

    if not q_words:
        return 0

    overlap = len(q_words & d_words)
    return overlap / len(q_words)

# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def root():
    return {"status": "SAP TCode API running", "loaded_tcodes": len(TCODES)}


@app.post("/search-tcode")
def search_tcode(req: SearchRequest):
    if not TCODES:
        raise HTTPException(status_code=500, detail="TCode dataset not loaded")

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    matches = []
    for row in TCODES:
        s = score_match(query, row["description"])
        if s > 0:
            matches.append({
                "tcode": row["tcode"],
                "description": row["description"],
                "module": row["module"],
                "score": round(s, 3)
            })

    matches.sort(key=lambda x: x["score"], reverse=True)
    top = matches[:5]

    return {
        "query": query,
        "count": len(top),
        "results": top
    }