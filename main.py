from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import os

app = FastAPI()

DB_PATH = "sap_tcodes.db"


# -----------------------------
# Request Schema
# -----------------------------
class SearchRequest(BaseModel):
    query: str


# -----------------------------
# DB Connection
# -----------------------------
def get_db():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Search T-Code
# -----------------------------
@app.post("/search-tcode")
def search_tcode(req: SearchRequest):

    query = req.query.lower().strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    conn = get_db()
    cur = conn.cursor()

    # --------------------------------------------------
    # 1️⃣ Alias Search
    # --------------------------------------------------
    cur.execute(
        "SELECT tcode, canonical_desc FROM aliases WHERE alias = ?",
        (query,)
    )
    row = cur.fetchone()

    if row:
        return {
            "source": "alias",
            "confidence": 0.95,
            "results": [
                {
                    "tcode": row["tcode"],
                    "description": row["canonical_desc"],
                    "module": "Unknown"
                }
            ]
        }

    # --------------------------------------------------
    # 2️⃣ FTS Search (MAIN SEARCH)
    # --------------------------------------------------
    cur.execute("""
        SELECT tcode, description, module
        FROM tcodes_fts
        WHERE tcodes_fts MATCH ?
        LIMIT 5
    """, (query,))

    rows = cur.fetchall()

    if rows:
        return {
            "source": "fts",
            "confidence": 0.85,
            "results": [
                {
                    "tcode": r["tcode"],
                    "description": r["description"],
                    "module": r["module"]
                }
                for r in rows
            ]
        }

    # --------------------------------------------------
    # 3️⃣ Fallback (Always valid JSON)
    # --------------------------------------------------
    return {
        "source": "fallback",
        "confidence": 0.3,
        "results": [],
        "message": "No confirmed ECC transaction found"
    }
