from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import sqlite3
import csv
import io
import requests

DB_PATH = "sap_tcodes.db"

app = FastAPI(
    title="SAP T-Code Lookup API",
    version="1.0.0",
    description="SAP ECC T-Code lookup with alias intelligence and fallback search"
)

# --------------------------------------------------
# Database
# --------------------------------------------------

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tcodes (
            tcode TEXT PRIMARY KEY,
            description TEXT,
            module TEXT
        )
    """)

    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS tcodes_fts
        USING fts5(tcode, description, module)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS aliases (
            alias TEXT PRIMARY KEY,
            tcode TEXT,
            canonical_desc TEXT
        )
    """)

    conn.commit()
    conn.close()

@app.on_event("startup")
def startup():
    init_db()

# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Search T-Code (Alias → FTS → Fallback)
# --------------------------------------------------

@app.post("/search-tcode")
def search_tcode(payload: dict):
    query = payload.get("query", "").lower().strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    conn = get_db()
    cur = conn.cursor()

    # 1️⃣ Alias lookup
    cur.execute(
        "SELECT tcode, canonical_desc FROM aliases WHERE alias = ?",
        (query,)
    )
    row = cur.fetchone()
    if row:
        return {
            "source": "alias",
            "confidence": 0.95,
            "tcode": row[0],
            "description": row[1]
        }

    # 2️⃣ FTS lookup
    cur.execute("""
        SELECT tcode, description, module
        FROM tcodes_fts
        WHERE tcodes_fts MATCH ?
        LIMIT 3
    """, (query,))
    rows = cur.fetchall()

    if rows:
        return {
            "source": "fts",
            "confidence": 0.85,
            "results": [
                {"tcode": r[0], "description": r[1], "module": r[2]}
                for r in rows
            ]
        }

    # 3️⃣ Fallback ECC community search
    fallback = ecc_community_search(query)
    return fallback


# --------------------------------------------------
# ECC Community Fallback (EXPOSED IN DOCS)
# --------------------------------------------------

@app.get(
    "/fallback/ecc-community-search",
    tags=["Fallback"],
    summary="ECC-only SAP Community fallback search"
)
def ecc_community_search(
    query: str = Query(..., description="Business task, e.g. 'reverse goods receipt'")
):
    search_q = f"site:community.sap.com {query} SAP ECC tcode -S/4HANA -HANA"

    try:
        r = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": search_q},
            timeout=10
        )
    except Exception:
        return {
            "found": False,
            "confidence": 0.2,
            "message": "Fallback search failed"
        }

    text = r.text.upper()
    known_ecc = ["MBST", "MIGO", "FB08", "VL09"]
    matches = [t for t in known_ecc if t in text]

    if not matches:
        return {
            "found": False,
            "confidence": 0.25,
            "message": "No ECC t-code found"
        }

    return {
        "found": True,
        "confidence": 0.55,
        "candidates": matches,
        "source": "SAP Community (ECC only)"
    }

# --------------------------------------------------
# Admin: Upload Alias CSV
# --------------------------------------------------

@app.post("/admin/upload-aliases", tags=["Admin"])
def upload_aliases(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV required")

    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    required = {"alias", "tcode", "canonical_desc"}
    if not required.issubset(reader.fieldnames):
        raise HTTPException(
            status_code=400,
            detail="CSV must contain: alias,tcode,canonical_desc"
        )

    conn = get_db()
    cur = conn.cursor()
    count = 0

    for row in reader:
        cur.execute("""
            INSERT OR REPLACE INTO aliases(alias, tcode, canonical_desc)
            VALUES (?, ?, ?)
        """, (
            row["alias"].strip().lower(),
            row["tcode"].strip().upper(),
            row["canonical_desc"].strip()
        ))
        count += 1

    conn.commit()
    conn.close()

    return {"status": "success", "rows_loaded": count}

# --------------------------------------------------
# Admin: Upload T-Code CSV
# --------------------------------------------------

@app.post("/admin/upload-tcodes", tags=["Admin"])
def upload_tcodes(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV required")

    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    required = {"tcode", "description", "module"}
    if not required.issubset(reader.fieldnames):
        raise HTTPException(
            status_code=400,
            detail="CSV must contain: tcode,description,module"
        )

    conn = get_db()
    cur = conn.cursor()
    count = 0

    for row in reader:
        tcode = row["tcode"].strip().upper()
        desc = row["description"].strip()
        mod = row["module"].strip()

        cur.execute("""
            INSERT OR REPLACE INTO tcodes(tcode, description, module)
            VALUES (?, ?, ?)
        """, (tcode, desc, mod))

        cur.execute("""
            INSERT INTO tcodes_fts(tcode, description, module)
            VALUES (?, ?, ?)
        """, (tcode, desc, mod))

        count += 1

    conn.commit()
    conn.close()

    return {"status": "success", "rows_loaded": count}
