from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from typing import Dict, Any

app = FastAPI(title="SAP TCode Backend")

# Allow calls from VAPI / web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "sap_tcodes.db"

# --------------------------------------------------
# DB Connection
# --------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --------------------------------------------------
# Health Check
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Dummy ECC fallback (safe placeholder)
# --------------------------------------------------

def ecc_community_search(query: str) -> Dict[str, Any]:
    return {
        "source": "fallback",
        "confidence": 0.4,
        "results": [],
        "message": "No confirmed ECC transaction found"
    }

# --------------------------------------------------
# Search T-Code (Alias → FTS → LIKE → Fallback)
# --------------------------------------------------
@app.post("/search-tcode")
def search_tcode(payload: Dict[str, Any]):
    query = payload.get("query", "").lower().strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    conn = get_db()
    cur = conn.cursor()

    # 1️⃣ Alias lookup
    cur.execute(
        "SELECT tcode, canonical_desc FROM aliases WHERE lower(alias)=?",
        (query,)
    )
    row = cur.fetchone()
    if row:
        return {
            "source": "alias",
            "confidence": 0.95,
            "results": [
                {
                    "tcode": row[0],
                    "description": row[1],
                    "module": ""
                }
            ]
        }

    # 2️⃣ FTS lookup
    try:
        cur.execute("""
            SELECT tcode, description, module
            FROM tcodes_fts
            WHERE tcodes_fts MATCH ?
            LIMIT 3
        """, (query,))
        rows = cur.fetchall()
    except Exception:
        rows = []

    if rows:
        return {
            "source": "fts",
            "confidence": 0.9,
            "results": [
                {"tcode": r[0], "description": r[1], "module": r[2]}
                for r in rows
            ]
        }

    # 3️⃣ Semantic LIKE search (fixes natural language queries)
    like_query = "%" + "%".join(query.split()) + "%"

    cur.execute("""
        SELECT tcode, description, module
        FROM tcodes
        WHERE lower(description) LIKE ?
        LIMIT 5
    """, (like_query,))

    rows = cur.fetchall()

    if rows:
        return {
            "source": "semantic",
            "confidence": 0.8,
            "results": [
                {"tcode": r[0], "description": r[1], "module": r[2]}
                for r in rows
            ]
        }

    # 4️⃣ ECC fallback
    return ecc_community_search(query)
