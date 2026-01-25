from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import sqlite3
import csv
import io
import os
from typing import Optional, List

# -------------------------
# App Init
# -------------------------

app = FastAPI(
    title="SAP T-Code Lookup API",
    version="1.0.0",
    description="Fast SAP T-code search with FTS5 + alias intelligence"
)

DB_PATH = "sap_tcodes.db"

# -------------------------
# Database Helpers
# -------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# -------------------------
# Models
# -------------------------

class SearchRequest(BaseModel):
    query: str
    module: Optional[str] = None

# -------------------------
# Health
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Alias Normalization
# -------------------------

def normalize_query(text: str) -> str:
    conn = get_db()
    cur = conn.cursor()

    words = text.lower().split()
    expanded = []

    for w in words:
        cur.execute(
            "SELECT canonical FROM aliases WHERE alias = ?",
            (w,)
        )
        row = cur.fetchone()
        expanded.append(row["canonical"] if row else w)

    return " ".join(expanded)

# -------------------------
# Search Endpoint
# -------------------------

@app.post("/search-tcode")
def search_tcode(req: SearchRequest):
    conn = get_db()
    cur = conn.cursor()

    normalized = normalize_query(req.query)

    sql = """
    SELECT tcode, description, module,
           bm25(sap_tcodes_fts) AS score
    FROM sap_tcodes_fts
    WHERE sap_tcodes_fts MATCH ?
    """
    params = [normalized]

    if req.module:
        sql += " AND module = ?"
        params.append(req.module)

    sql += " ORDER BY score LIMIT 5"

    cur.execute(sql, params)
    rows = cur.fetchall()

    if rows:
        return {
            "normalized_query": normalized,
            "results": [dict(r) for r in rows],
            "confidence": "high" if rows[0]["score"] < -5 else "medium"
        }

    # Alias suggestion fallback
    cur.execute(
        "SELECT DISTINCT canonical FROM aliases WHERE canonical LIKE ? LIMIT 3",
        (f"%{normalized}%",)
    )
    suggestions = [r["canonical"] for r in cur.fetchall()]

    return {
        "normalized_query": normalized,
        "results": [],
        "suggestions": suggestions,
        "confidence": "low"
    }

# -------------------------
# Admin: Upload Alias CSV
# -------------------------

@app.post("/admin/upload-aliases", tags=["Admin"])
def upload_aliases(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV required")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS aliases (
            alias TEXT PRIMARY KEY,
            canonical TEXT
        )
    """)

    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    for row in reader:
        cur.execute(
            "INSERT OR REPLACE INTO aliases (alias, canonical) VALUES (?, ?)",
            (row["alias"].lower(), row["canonical"].lower())
        )

    conn.commit()
    return {"status": "aliases uploaded"}

# -------------------------
# Admin: Upload T-code CSV
# -------------------------

@app.post("/admin/upload-tcodes", tags=["Admin"])
def upload_tcodes(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV required")

    conn = get_db()
    cur = conn.cursor()

    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    cur.execute("DELETE FROM sap_tcodes")

    for row in reader:
        cur.execute(
            """
            INSERT INTO sap_tcodes (tcode, description, module)
            VALUES (?, ?, ?)
            """,
            (row["tcode"], row["description"], row["module"])
        )

    # Rebuild FTS
    cur.execute("DELETE FROM sap_tcodes_fts")
    cur.execute("""
        INSERT INTO sap_tcodes_fts (tcode, description, module)
        SELECT tcode, description, module FROM sap_tcodes
    """)

    conn.commit()
    return {"status": "tcodes uploaded & FTS synced"}
