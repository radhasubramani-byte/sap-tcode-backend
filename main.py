from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import sqlite3
import csv
from io import TextIOWrapper
from typing import Optional

DB_PATH = "sap_tcodes.db"

app = FastAPI(
    title="SAP T-Code Lookup API",
    version="1.0.0"
)

# -------------------------
# Database helpers
# -------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()

    # Main tcode table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tcodes (
            tcode TEXT PRIMARY KEY,
            description TEXT,
            module TEXT
        )
    """)

    # FTS table
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS tcodes_fts
        USING fts5(tcode, description, module)
    """)

    # Alias table (UPDATED SCHEMA)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS aliases (
            alias TEXT PRIMARY KEY,
            tcode TEXT NOT NULL,
            canonical_desc TEXT
        )
    """)

    conn.commit()

init_db()

# -------------------------
# Health
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Search API
# -------------------------

class SearchRequest(BaseModel):
    query: str
    module: Optional[str] = None

@app.post("/search-tcode")
def search_tcode(req: SearchRequest):
    conn = get_db()
    cur = conn.cursor()

    q = req.query.strip().lower()

    # 1️⃣ Alias lookup first
    cur.execute("""
        SELECT tcode, canonical_desc
        FROM aliases
        WHERE alias = ?
    """, (q,))
    alias_hit = cur.fetchone()

    if alias_hit:
        return {
            "confidence": 0.95,
            "source": "alias",
            "results": [{
                "tcode": alias_hit["tcode"],
                "description": alias_hit["canonical_desc"]
            }]
        }

    # 2️⃣ FTS fallback
    if req.module:
        cur.execute("""
            SELECT tcode, description, module
            FROM tcodes_fts
            WHERE tcodes_fts MATCH ?
              AND module = ?
            LIMIT 5
        """, (q, req.module))
    else:
        cur.execute("""
            SELECT tcode, description, module
            FROM tcodes_fts
            WHERE tcodes_fts MATCH ?
            LIMIT 5
        """, (q,))

    rows = cur.fetchall()

    if not rows:
        return {
            "confidence": 0.0,
            "message": "No matching SAP T-code found"
        }

    return {
        "confidence": 0.75,
        "source": "fts",
        "results": [
            {
                "tcode": r["tcode"],
                "description": r["description"],
                "module": r["module"]
            } for r in rows
        ]
    }

# -------------------------
# Admin: Upload Alias CSV
# -------------------------

@app.post("/admin/upload-aliases", tags=["Admin"])
def upload_aliases(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV file required")

    conn = get_db()
    cur = conn.cursor()

    reader = csv.DictReader(TextIOWrapper(file.file, encoding="utf-8"))

    required_cols = {"alias", "tcode", "canonical_desc"}
    if not required_cols.issubset(reader.fieldnames):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain columns: {required_cols}"
        )

    count = 0
    for row in reader:
        cur.execute("""
            INSERT OR REPLACE INTO aliases (alias, tcode, canonical_desc)
            VALUES (?, ?, ?)
        """, (
            row["alias"].strip().lower(),
            row["tcode"].strip().upper(),
            row["canonical_desc"].strip()
        ))
        count += 1

    conn.commit()
    return {"status": "success", "aliases_loaded": count}

# -------------------------
# Admin: Upload T-code CSV
# -------------------------

@app.post("/admin/upload-tcodes", tags=["Admin"])
def upload_tcodes(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV file required")

    conn = get_db()
    cur = conn.cursor()

    reader = csv.DictReader(TextIOWrapper(file.file, encoding="utf-8"))

    required_cols = {"tcode", "description", "module"}
    if not required_cols.issubset(reader.fieldnames):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain columns: {required_cols}"
        )

    count = 0
    for row in reader:
        tcode = row["tcode"].strip().upper()
        desc = row["description"].strip()
