import pandas as pd
from sqlalchemy import create_engine, text

CSV_PATH = "sap_tcodes.csv"
DB_PATH = "sqlite:///sap_tcodes.db"

engine = create_engine(DB_PATH)

print("🔹 Loading CSV...")
df = pd.read_csv(CSV_PATH)

# Validate
if not {"tcode", "description"}.issubset(df.columns):
    raise ValueError("CSV must contain tcode and description columns")

# Normalize
df["tcode"] = df["tcode"].astype(str).str.upper()
df["description"] = df["description"].astype(str)

if "module" not in df.columns:
    df["module"] = ""
else:
    df["module"] = df["module"].astype(str).str.upper()

# 👉 IMPORTANT PART: use RAW sqlite connection for pandas
raw_conn = engine.raw_connection()
try:
    # 1️⃣ Create / replace main table
    df.to_sql("sap_tcodes", raw_conn, if_exists="replace", index=False)

    cursor = raw_conn.cursor()

    # 2️⃣ Create FTS5 table
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS sap_tcodes_fts
        USING fts5(tcode, description, module);
    """)

    # 3️⃣ Rebuild FTS index
    cursor.execute("DELETE FROM sap_tcodes_fts;")
    cursor.execute("""
        INSERT INTO sap_tcodes_fts (tcode, description, module)
        SELECT tcode, description, module FROM sap_tcodes;
    """)

    raw_conn.commit()

finally:
    raw_conn.close()

print("✅ sap_tcodes.db created with FTS5 enabled")
