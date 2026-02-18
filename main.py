from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import re

app = FastAPI(title="SAP ECC TCode Search API")

# --------------------------------------------------
# CORS (important for VAPI + external callers)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# DATABASE
# --------------------------------------------------
DB_PATH = "sap_tcodes.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# NORMALIZE USER LANGUAGE → SAFE FTS QUERY
# Prevents SQLite MATCH crashes
# --------------------------------------------------
def normalize_query(text: str) -> str:

    text = text.lower().strip()

    # remove characters that break FTS parser
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    words = [w for w in text.split() if len(w) > 2]

    if not words:
        return text

    # OR search improves recall dramatically
    return " OR ".join(words)


# --------------------------------------------------
# ECC COMMUNITY FALLBACK
# Always return SAFE schema
# --------------------------------------------------
def ecc_community_search(query: str):

    # minimal safe fallback
    return {
        "source": "fallback",
        "confidence": 0.40,
        "results": [],
        "message": "No confirmed ECC transaction found"
    }


# --------------------------------------------------
# MAIN SEARCH ENDPOINT
# Alias → FTS → Fallback
# --------------------------------------------------
@app.post("/search-tcode")
def search_tcode(payload: dict):

    query = payload.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    conn = get_db()
    cur = conn.cursor()

    clean_query = query.lower().strip()

    # ---------------------------
    # 1️⃣ Alias Lookup
    # ---------------------------
    try:
        cur.execute(
            "SELECT tcode, canonical_desc FROM aliases WHERE alias = ?",
            (clean_query,)
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
                        "module": "General"
                    }
                ]
            }

    except Exception as e:
        print("Alias search error:", e)


    # ---------------------------
    # 2️⃣ FTS SEARCH (SAFE)
    # ---------------------------
    try:
        fts_query = normalize_query(clean_query)

        cur.execute("""
            SELECT tcode, description, module
            FROM tcodes_fts
            WHERE tcodes_fts MATCH ?
            LIMIT 5
        """, (fts_query,))

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

    except Exception as e:
        print("FTS FAILED:", e)


    # ---------------------------
    # 3️⃣ FALLBACK
    # ---------------------------
    return ecc_community_search(clean_query)
