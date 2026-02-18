from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import os

app = FastAPI(title="SAP TCode Backend")

DB_PATH = "sap_tcodes.db"

# -------------------------

# Request Model

# -------------------------

class SearchRequest(BaseModel):
query: str

# -------------------------

# Database Connection

# -------------------------

def get_db():
if not os.path.exists(DB_PATH):
raise HTTPException(status_code=500, detail="Database not found")

```
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
return conn
```

# -------------------------

# Health Check

# -------------------------

@app.get("/health")
def health():
return {"status": "ok"}

# -------------------------

# SEARCH TCODE

# -------------------------

@app.post("/search-tcode")
def search_tcode(req: SearchRequest):
query = req.query.lower().strip()

```
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
        "tcode": row["tcode"],
        "description": row["canonical_desc"],
        "module": "MM"
    }

# 2️⃣ Description LIKE search (robust for voice)
cur.execute("""
    SELECT tcode, description, module
    FROM tcodes
    WHERE lower(description) LIKE ?
    LIMIT 5
""", (f"%{query}%",))

rows = cur.fetchall()

if rows:
    return {
        "source": "database",
        "confidence": 0.9,
        "results": [
            {
                "tcode": r["tcode"],
                "description": r["description"],
                "module": r["module"]
            }
            for r in rows
        ]
    }

# 3️⃣ Nothing found
return {
    "source": "fallback",
    "confidence": 0.3,
    "results": [],
    "message": "No ECC transaction found"
}
```
