from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from typing import Optional, Dict, Any

app = FastAPI(title="SAP TCode Backend")

# -----------------------------
# CORS (required for VAPI + web)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "sap_tcodes.db"


# --------------------------------------------------
# Database connection
# --------------------------------------------------
def get_db():
    return sqlite3.connect(DB_PATH)


# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# ECC fallback (safe stub if none found)
# Replace later if you want community scraping
# --------------------------------------------------
def ecc_community_search(query: str) -> Optional[Dict[str, Any]]:
    # You can later plug your ECC search here
    return None


# --------------------------------------------------
# Search T-Code (Alias → FTS → Fallback)
# CONSISTENT RESPONSE STRUCTURE (CRITICAL FOR VAPI)
# --------------------------------------------------
@app.post("/search-tcode")
def search_tcode(payload: dict):
    query = payload.get("query", "").lower().strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    conn = get_db()
    cur = conn.cursor()

    # --------------------------------------------------
    # 1️⃣ Alias lookup (highest confidence)
    # --------------------------------------------------
    cur.execute(
        "SELECT tcode, canonical_desc FROM aliases WHERE lower(alias) = ?",
        (query,)
    )
    row = cur.fetchone()

    if row:
        conn.close()
        return {
            "found": True,
            "tcode": row[0],
            "description": row[1],
            "module": "Unknown",
            "confidence": 0.95,
            "source": "alias",
            "alternatives": []
        }

    # --------------------------------------------------
    # 2️⃣ Full Text Search (main database)
    # --------------------------------------------------
    try:
        cur.execute("""
            SELECT tcode, description, module
            FROM tcodes_fts
            WHERE tcodes_fts MATCH ?
            LIMIT 5
        """, (query,))
        rows = cur.fetchall()
    except Exception:
        rows = []

    if rows:
        best = rows[0]
        conn.close()
        return {
            "found": True,
            "tcode": best[0],
            "description": best[1],
            "module": best[2],
            "confidence": 0.85,
            "source": "search",
            "alternatives": [
                {"tcode": r[0], "description": r[1], "module": r[2]}
                for r in rows[1:]
            ]
        }

    # --------------------------------------------------
    # 3️⃣ ECC fallback search
    # --------------------------------------------------
    fallback = ecc_community_search(query)

    if fallback:
        conn.close()
        return {
            "found": True,
            "tcode": fallback.get("tcode"),
            "description": fallback.get("description"),
            "module": fallback.get("module", "ECC"),
            "confidence": 0.55,
            "source": "fallback",
            "alternatives": []
        }

    # --------------------------------------------------
    # 4️⃣ Nothing found
    # --------------------------------------------------
    conn.close()
    return {
        "found": False,
        "tcode": None,
        "description": None,
        "module": None,
        "confidence": 0.0,
        "source": "none",
        "alternatives": []
    }
