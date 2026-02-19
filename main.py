from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3

app = FastAPI(title="SAP ECC TCode API")

DB_PATH = "sap_tcodes.db"

-----------------------------
Request Model
-----------------------------

class SearchRequest(BaseModel):
query: str

-----------------------------
Database
-----------------------------

def get_db():
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
return conn

-----------------------------
Health
-----------------------------

@app.get("/health")
def health():
return {"status": "ok"}

-----------------------------
Search TCode
-----------------------------

@app.post("/search-tcode")
def search_tcode(req: SearchRequest):

query = req.query.lower().strip()
if not query:
    raise HTTPException(status_code=400, detail="query required")

conn = get_db()
cur = conn.cursor()

# 1️⃣ Alias search
cur.execute(
    "SELECT tcode, canonical_desc FROM aliases WHERE alias = ?",
    (query,)
)
row = cur.fetchone()
if row:
    return {
        "found": True,
        "source": "alias",
        "confidence": 0.95,
        "tcode": row["tcode"],
        "description": row["canonical_desc"],
        "module": None,
        "alternatives": []
    }

# 2️⃣ FTS search
cur.execute(
    """
    SELECT tcode, description, module
    FROM tcodes_fts
    WHERE tcodes_fts MATCH ?
    LIMIT 5
    """,
    (query,)
)
rows = cur.fetchall()

if rows:
    best = rows[0]
    alternatives = []

    for r in rows[1:]:
        alternatives.append({
            "tcode": r["tcode"],
            "description": r["description"],
            "module": r["module"]
        })

    return {
        "found": True,
        "source": "fts",
        "confidence": 0.85,
        "tcode": best["tcode"],
        "description": best["description"],
        "module": best["module"],
        "alternatives": alternatives
    }

# 3️⃣ Nothing found
return {
    "found": False,
    "source": "none",
    "confidence": 0.0,
    "tcode": None,
    "description": None,
    "module": None,
    "alternatives": []
}