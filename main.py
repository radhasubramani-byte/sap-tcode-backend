from fastapi import FastAPI, HTTPException
import sqlite3
import os
import csv

app = FastAPI(title="SAP TCode Search API")

DB_PATH = "sap_tcodes.db"
CSV_PATH = "tcodes.csv"


# --------------------------------------------------
# DATABASE INITIALIZATION (RUNS ON STARTUP)
# --------------------------------------------------
def init_db():

    first_time = not os.path.exists(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Main table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tcodes (
        tcode TEXT PRIMARY KEY,
        description TEXT,
        module TEXT
    )
    """)

    # Alias table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS aliases (
        alias TEXT PRIMARY KEY,
        tcode TEXT,
        canonical_desc TEXT
    )
    """)

    # FTS search index
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS tcodes_fts USING fts5(
        tcode, description, module
    )
    """)

    # If empty → load CSV
    cur.execute("SELECT COUNT(*) FROM tcodes")
    count = cur.fetchone()[0]

    if count == 0:
        print("Loading TCode CSV into database...")

        with open(CSV_PATH, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                tcode, desc, module = row[0].strip(), row[1].strip(), row[2].strip()
                cur.execute("INSERT OR IGNORE INTO tcodes VALUES (?, ?, ?)", (tcode, desc, module))
                cur.execute("INSERT INTO tcodes_fts VALUES (?, ?, ?)", (tcode, desc, module))

        # Build aliases automatically
        cur.execute("""
        INSERT OR IGNORE INTO aliases
        SELECT LOWER(description), tcode, description FROM tcodes
        """)

        print("Database initialized successfully")

    conn.commit()
    conn.close()


@app.on_event("startup")
def startup():
    init_db()


# --------------------------------------------------
# SEARCH API
# --------------------------------------------------
@app.post("/search-tcode")
def search_tcode(payload: dict):

    query = payload.get("query", "").lower().strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1️⃣ Alias lookup (HIGH CONFIDENCE)
    cur.execute(
        "SELECT tcode, canonical_desc FROM aliases WHERE alias = ?",
        (query,)
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return {
            "match_type": "exact",
            "confidence": 0.95,
            "tcode": row[0],
            "description": row[1],
            "module": None
        }

    # 2️⃣ FTS search
    cur.execute("""
        SELECT tcode, description, module
        FROM tcodes_fts
        WHERE tcodes_fts MATCH ?
        LIMIT 3
    """, (query,))
    rows = cur.fetchall()

    conn.close()

    if rows:
        return {
            "match_type": "search",
            "confidence": 0.85,
            "results": [
                {"tcode": r[0], "description": r[1], "module": r[2]}
                for r in rows
            ]
        }

    # 3️⃣ No result
    return {
        "match_type": "none",
        "confidence": 0.0,
        "results": []
    }
