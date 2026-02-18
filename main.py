from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import re

app = FastAPI()

# Allow VAPI & browser

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

DB_PATH = "sap_tcodes.db"

# ---------------- DB ----------------

def get_db():
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
return conn

# ---------------- NLP CLEANING ----------------

STOP_WORDS = {
"i","want","to","the","a","an","please","today",
"me","show","how","do","make","create","display"
}

def normalize_query(text: str):
text = text.lower()
text = re.sub(r"[^a-z0-9\s]", " ", text)
tokens = [t for t in text.split() if t not in STOP_WORDS]
return tokens

def build_fts_query(tokens):
if not tokens:
return ""
return " OR ".join(tokens)

# ---------------- HEALTH ----------------

@app.get("/health")
def health():
return {"status": "ok"}

# ---------------- SEARCH TCODE ----------------

@app.post("/search-tcode")
def search_tcode(payload: dict):
query = payload.get("query", "").strip()
if not query:
raise HTTPException(status_code=400, detail="query required")

```
conn = get_db()
cur = conn.cursor()

# 1️⃣ Alias match
cur.execute(
    "SELECT tcode, canonical_desc FROM aliases WHERE lower(alias)=?",
    (query.lower(),)
)
row = cur.fetchone()
if row:
    return {
        "status": "found",
        "confidence": 0.95,
        "source": "alias",
        "results": [
            {
                "tcode": row["tcode"],
                "description": row["canonical_desc"],
                "module": None
            }
        ]
    }

# 2️⃣ Clean query for FTS
tokens = normalize_query(query)
fts_query = build_fts_query(tokens)

try:
    if fts_query:
        cur.execute("""
            SELECT tcode, description, module
            FROM tcodes_fts
            WHERE tcodes_fts MATCH ?
            LIMIT 5
        """, (fts_query,))
        rows = cur.fetchall()
    else:
        rows = []
except Exception:
    rows = []

if rows:
    return {
        "status": "found",
        "confidence": 0.85,
        "source": "fts",
        "results": [
            {
                "tcode": r["tcode"],
                "description": r["description"],
                "module": r["module"]
            }
            for r in rows
        ]
    }

# 3️⃣ LIKE fallback (VERY IMPORTANT — catches ME21N cases)
like_query = "%" + "%".join(tokens) + "%"
cur.execute("""
    SELECT tcode, description, module
    FROM tcodes
    WHERE lower(description) LIKE ?
    LIMIT 5
""", (like_query,))
rows = cur.fetchall()

if rows:
    return {
        "status": "found",
        "confidence": 0.7,
        "source": "like",
        "results": [
            {
                "tcode": r["tcode"],
                "description": r["description"],
                "module": r["module"]
            }
            for r in rows
        ]
    }

# 4️⃣ Final fallback
return {
    "status": "not_found",
    "confidence": 0.3,
    "source": "fallback",
    "results": [],
    "message": "No confirmed ECC transaction found"
}
```
