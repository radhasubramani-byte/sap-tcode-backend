from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import sqlite3
import csv
import io
import os

DB_PATH = "sap_tcodes.db"

app = FastAPI(title="SAP T-Code Backend with FTS5")

# ---------- DB Helpers ----------

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def tokenize(query: str):
    return " OR ".join(query.lower().split())

# ---------- Models ----------

class SearchRequest(BaseModel):
    query: str
    limit: int = 3

# ---------- Health ----------

@app.get("/")
def root():
    return {"status": "ok"}

# ---------- SEARCH (VAPI uses this) ----------

@app.post("/search-tcode")
def search_tcode(req: SearchRequest):
    if not req.query.strip():
        return {"results": []}

    conn = get_db()
    cur = conn.cursor()

    fts_query = tokenize(req.query)

    cur.execute(
        """
        SELECT tcode, description
        FROM sap_tcodes_fts
        WHERE sap_tcodes_fts MATCH ?
        LIMIT ?
        """,
        (fts_query, req.limit),
    )

    rows = cur.fetchall()
    conn.close()

    # ALWAYS return valid JSON
    results = [
        {"tcode": r["tcode"], "description": r["description"]}
        for r in rows
    ]

    return {"results": results}

# ---------- ADMIN CSV UPLOAD (LIVE FTS SYNC) ----------

@app.post("/admin/upload-csv")
def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV only")

    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    conn = get_db()
    cur = conn.cursor()

    # Clear base table
    cur.execute("DELETE FROM sap_tcodes")

    rows = 0
    for row in reader:
        cur.execute(
            "INSERT INTO sap_tcodes (tcode, description) VALUES (?, ?)",
            (row["tcode"], row["description"]),
        )
        rows += 1

    # 🔥 LIVE FTS REBUILD
    cur.execute("DELETE FROM sap_tcodes_fts")
    cur.execute(
        """
        INSERT INTO sap_tcodes_fts (rowid, tcode, description)
        SELECT rowid, tcode, description FROM sap_tcodes
        """
    )

    conn.commit()
    conn.close()

    return {
        "status": "uploaded",
        "rows": rows,
        "fts": "synced"
    }
