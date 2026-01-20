import os
from fastapi import FastAPI, UploadFile, Depends, Header, HTTPException
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# -------------------
# Environment setup
# -------------------
load_dotenv()
ADMIN_API_KEY = os.getenv("SAP_ADMIN_KEY")

# -------------------
# FastAPI app
# -------------------
app = FastAPI(title="SAP T-code Backend", version="1.0")
engine = create_engine("sqlite:///sap_tcodes.db")

# -------------------
# Security dependency
# -------------------
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -------------------
# Request models
# -------------------
class SearchRequest(BaseModel):
    query: str

# -------------------
# Public endpoint (used by VAPI)
# -------------------
@app.post("/search-tcode")
def search_tcode(req: SearchRequest, module: str | None = None):
    with engine.connect() as conn:
        sql = """
        SELECT tcode, description, module
        FROM sap_tcodes
        WHERE description LIKE :q
        """
        params = {"q": f"%{req.query}%"}

        if module:
            sql += " AND module = :m"
            params["m"] = module.upper()

        sql += " ORDER BY description LIMIT 5"

        rows = conn.execute(text(sql), params).fetchall()

    if not rows:
        return {"found": False}

    modules = {r[2] for r in rows if r[2]}
    ambiguous = len(modules) > 1

    return {
        "found": True,
        "results": [
            {"tcode": r[0], "description": r[1], "module": r[2]}
            for r in rows
        ],
        "ambiguous_modules": ambiguous
    }

# -------------------
# Admin: CSV upload (API key protected)
# -------------------
@app.post("/admin/upload-csv")
def upload_csv(file: UploadFile, _: None = Depends(verify_api_key)):
    df = pd.read_csv(file.file)

    required_cols = {"tcode", "description"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail="CSV must contain at least tcode and description columns"
        )

    if "module" not in df.columns:
        df["module"] = None

    df["tcode"] = df["tcode"].str.upper()
    df["module"] = df["module"].str.upper()

    raw_conn = engine.raw_connection()
    try:
        df.to_sql(
            "sap_tcodes",
            raw_conn,
            if_exists="append",
            index=False
        )
    finally:
        raw_conn.close()

    return {
        "status": "success",
        "rows_added": len(df)
    }

# -------------------
# OpenAI / suggest-module endpoint intentionally DISABLED
# -------------------

# -------------------
# Run app (Render-compatible)
# -------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
