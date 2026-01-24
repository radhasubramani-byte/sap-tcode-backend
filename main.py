from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import sqlite3
import csv
import io
import os

DB_PATH = "sap_tcodes.db"

app = FastAPI(
    title="SAP T-Code Lookup API",
    description="Fast SAP T-code search using SQLite FTS5 with alias intelligence",
    version="1.1.0",
)

# -------------------------
# DB Helpers
# -------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def normalize(text: str) -> str:
    return text.lower().strip()


# -------------------------
# Startup: Ensure tables
# -------------------------

@app.on_event("startup")
def startup():
    conn = get_db()
    cur = conn.cursor()

    # Main table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sap_tcodes (
            tcode TEXT PRIMARY KEY,
            description TEXT,
            module TEXT
        )
    """)

    # FTS table
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS sap_tcodes_fts
        USING fts5(tcode, description, module)
    """)

    # Alias table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sap_aliases (
            alias TEXT PRIMARY KEY,
            canonical TEXT
        )
    """)

    conn.commit()
    conn.close()


# -------------------------
# Models
# -------------------------

class SearchRequest(BaseModel):
    query: str
    module: str | None = None


# -------------------------
# Search Logic
# -------------------------

@app.post("/search-tcode")
def search_tcode(req: SearchRequest):
    query = normalize(req.query)
    conn = get_db()
    cur = conn.cursor()

    # 1️⃣ Exact T-code match
    cur.execute(
        "SELECT * FROM sap_tcodes WHERE lower(tcode)=?",
        (query.upper(),)
    )
    row = cur.fetchone()
    if row:
        return {
            "results": [dict(row)],
            "confidence": 0.95,
            "match_type": "exact"
        }

    # 2️⃣ FTS search
    sql = """
        SELECT tcode, description, module
        FROM sap_tcodes_fts
        WHERE sap_tcodes_fts MATCH ?
        LIMIT 5
    """
    cur.execute(sql, (query,))
    rows = cur.fetchall()

    if rows:
        return {
            "results": [dict(r) for r in rows],
            "confidence": 0.80,
            "match_type": "fts"
        }

    # 3️⃣ Alias lookup
    cur.execute(
        "SELECT canonical FROM sap_aliases WHERE alias=?",
        (query,)
    )
    alias = cur.fetchone()

    if alias:
        canonical = alias["canonical"]
        cur.execute(
            "SELECT * FROM sap_tcodes WHERE lower(description) LIKE ?",
            (f"%{canonical.lower()}%",)
        )
        rows = cur.fetchall()

        if rows:
            return {
                "results": [dict(r) for r in rows],
                "confidence": 0.65,
                "match_type": "alias_suggestion",
                "suggestion": canonical
            }

    return {
        "results": [],
        "confidence": 0.0,
        "match_type": "none"
    }


# -------------------------
# Admin: Upload T-code CSV
# -------------------------

@app.post("/admin/upload-tcodes")
def upload_tcodes(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV required")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("DELETE FROM sap_tcodes")
    cur.execute("DELETE FROM sap_tcodes_fts")

    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    for row in reader:
        tcode = row["tcode"].strip()
        desc = row["description"].strip()
        module = row.get("module", "").strip()

        cur.execute(
            "INSERT INTO sap_tcodes VALUES (?, ?, ?)",
            (tcode, desc, module)
        )
        cur.execute(
            "INSERT INTO sap_tcodes_fts VALUES (?, ?, ?)",
            (tcode, desc, module)
        )

    conn.commit()
    conn.close()

    return {"status": "T-codes uploaded and FTS synced"}


# -------------------------
# Admin: Upload Alias CSV
# -------------------------

@app.post("/admin/upload-aliases")
def upload_aliases(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV required")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM sap_aliases")

    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    for row in reader:
        cur.execute(
            "INSERT INTO sap_aliases VALUES (?, ?)",
            (
                normalize(row["alias"]),
                normalize(row["canonical"])
            )
        )

    conn.commit()
    conn.close()

    return {"status": "Aliases uploaded successfully"}


# -------------------------
# Health
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}
