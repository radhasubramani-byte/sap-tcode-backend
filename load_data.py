import pandas as pd
from sqlalchemy import create_engine

# Load CSV
df = pd.read_csv("sap_tcodes.csv")

# Normalize
df["tcode"] = df["tcode"].str.upper()
df["module"] = df["module"].str.upper()

# Create SQLAlchemy engine
engine = create_engine("sqlite:///sap_tcodes.db")

# ✅ USE RAW DBAPI CONNECTION (this fixes it)
raw_conn = engine.raw_connection()

try:
    df.to_sql(
        "sap_tcodes",
        raw_conn,
        if_exists="replace",
        index=False
    )
finally:
    raw_conn.close()

print("✅ SAP T-codes loaded successfully")
