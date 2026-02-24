import pandas as pd
from app.services.embedding_service import build_vector_store

def load_knowledge():

    documents = []

    # ---------- TCodes ----------
    tcodes = pd.read_csv("knowledge/tcodes.csv")

    for _, row in tcodes.iterrows():
        doc = f"""
        TCode: {row['tcode']}
        Description: {row['description']}
        Module: {row['module']}
        Keywords: {row.get('keywords','')}
        """
        documents.append(doc)

    # ---------- Aliases ----------
    for file in ["mm_aliases.csv","sd_aliases.csv","le_aliases.csv"]:
        df = pd.read_csv(f"knowledge/{file}")

        for _, row in df.iterrows():
            doc = f"""
            User says: {row['alias']}
            Means: {row['meaning']}
            TCode: {row['tcode']}
            """
            documents.append(doc)

    build_vector_store(documents)