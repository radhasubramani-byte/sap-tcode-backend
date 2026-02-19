from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///sap_tcodes.db")

with engine.connect() as conn:
    # 1) Create FTS5 virtual table
    conn.execute(text("""
        CREATE VIRTUAL TABLE IF NOT EXISTS sap_tcodes_fts
        USING fts5(
            tcode,
            description,
            module
        );
    """))

    # 2) Clear existing FTS data (safe re-sync)
    conn.execute(text("DELETE FROM sap_tcodes_fts;"))

    # 3) Populate FTS from main table
    conn.execute(text("""
        INSERT INTO sap_tcodes_fts (tcode, description, module)
        SELECT tcode, description, module FROM sap_tcodes;
    """))

print("✅ FTS5 enabled and data synced")
