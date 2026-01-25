from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import sqlite3
import csv
import io
import os

DB_PATH = "sap_tcodes.db"

app = FastAPI(
    title="SAP T-Code Lookup API",
    version="1.0.0"
)

# -------------------------
# Database Helpers
# -------------------------

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_db()
    cur = conn.cursor()

    # T-code table
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

    # 🔥 RESET aliases table to avoid schema mismatch
    cur.execute("DROP TABLE IF EXISTS aliases")

    # Alias table (NEW SCHEMA)
    cur.execute("""
        CREATE TABLE aliases (
            alias TEXT PRIMARY KEY,
            tcode TEXT NOT NULL,
            canonical_desc TEXT
        )
    """)

    conn.commit()


init_db()

# -------------------------
# Models
# -------------------------

class SearchRequest(BaseModel):
    query: str
    module: str | None = None


# -------------------------
# Health
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Search Logic
# -------------------------

@app.post("/search-tcode")
def search_tcode(req: SearchRequest):
    query = req.query.lower().strip()
    conn = get_db()
    cur = conn.cursor()

    # 1️⃣ Alias lookup FIRST
    cur.execute("""
        SELECT tcode, canonical_desc
        FROM aliases
        WHERE alias = ?
    """, (query,))
    alias_hit = cur.fetchone()

    if alias_hit:
        return {
            "confidence": 0.95,
            "source": "alias",
            "results": [{
                "tcode": alias_hit[0],
                "description": alias_hit[1]
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
        """, (query, req.module))
    else:
        cur.execute("""
            SELECT tcode, description, module
            FROM tcodes_fts
            WHERE tcodes_fts MATCH ?
            LIMIT 5
        """, (query,))

    rows = cur.fetchall()

    if not rows:
        # 🔁 Alias suggestion fallback
        cur.execute("""
            SELECT alias, tcode, canonical_desc
            FROM aliases
            WHERE alias LIKE ?
            LIMIT 3
        """, (f"%{query}%",))
        suggestions = cur.fetchall()

        if suggestions:
            return {
                "confidence": 0.6,
                "suggestions": [
                    {
                        "alias": s[0],
                        "tcode": s[1],
                        "description": s[2]
                    } for s in suggestions
                ]
            }

        return {"confidence": 0.0, "results": []}

    return {
        "confidence": 0.85,
        "source": "fts",
        "results": [
            {
                "tcode": r[0],
                "description": r[1],
                "module": r[2]
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

    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    required_cols = {"alias", "tcode", "canonical_desc"}
    if not required_cols.issubset(reader.fieldnames):
        raise HTTPException(
            status_code=400,
            detail="CSV must contain: alias,tcode,canonical_desc"
        )

    conn = get_db()
    cur = conn.cursor()
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

    return {
        "status": "success",
        "aliases_loaded": count
    }


# -------------------------
# Admin: Upload T-code CSV
# -------------------------

@app.post("/admin/upload-tcodes", tags=["Admin"])
def upload_tcodes(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV file required")

    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    required_cols = {"tcode", "description", "module"}
    if not required_cols.issubset(reader.fieldnames):
        raise HTTPException(
            status_code=400,
            detail="CSV must contain: tcode,description,module"
        )

    conn = get_db()
    cur = conn.cursor()
    count = 0

    for row in reader:
        cur.execute("""
            INSERT OR REPLACE INTO tcodes (tcode, description, module)
            VALUES (?, ?, ?)
        """, (
            row["tcode"].strip().upper(),
            row["description"].strip(),
            row["module"].strip().upper()
        ))

        cur.execute("""
            INSERT INTO tcodes_fts (tcode, description, module)
            VALUES (?, ?, ?)
        """, (
            row["tcode"].strip().upper(),
            row["description"].strip(),
            row["module"].strip().upper()
        ))

        count += 1

    conn.commit()

    return {
        "status": "success",
        "tcodes_loaded": count
    }
