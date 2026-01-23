import os
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

DB_PATH = "sap_tcodes.db"

app = FastAPI(
    title="SAP T-Code Lookup API",
    description="Fast SAP T-code search using SQLite FTS5",
    version="1.0.0",
)

# ---------- Models ----------

class SearchRequest(BaseModel):
    query: str
    module: Optional[str] = None


# ---------- Helpers ----------

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------- Health ----------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Search ----------

@app.post("/search-tcode")
def search_tcode(req: SearchRequest):
    query = req.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        if req.module:
            sql = """
                SELECT tcode, description, module
                FROM sap_tcodes_fts
                WHERE sap_tcodes_fts MATCH ?
                  AND module = ?
                LIMIT 5
            """
            cur.execute(sql, (query, req.module.upper()))
        else:
            sql = """
                SELECT tcode, description, module
                FROM sap_tcodes_fts
                WHERE sap_tcodes_fts MATCH ?
                LIMIT 5
            """
            cur.execute(sql, (query,))

        rows = cur.fetchall()

    finally:
        conn.close()

    if not rows:
        return {
            "found": False,
            "message": "No matching SAP T-code found",
        }

    return {
        "found": True,
        "results": [
            {
                "tcode": row["tcode"],
                "description": row["description"],
                "module": row["module"],
            }
            for row in rows
        ],
    }
