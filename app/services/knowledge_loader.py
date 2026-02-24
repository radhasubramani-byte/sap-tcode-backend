# app/services/knowledge_loader.py

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

# -------------------------------------------------------
# Resolve absolute project paths (works in Docker/Render)
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


def _safe_read_csv(file_path: Path) -> pd.DataFrame:
    """Safely read CSV and raise clear error if missing."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"\n❌ DATA FILE MISSING: {file_path}\n"
            f"Make sure the file exists inside the /data folder in your repo.\n"
            f"Render path checked: {file_path}\n"
        )
    return pd.read_csv(file_path)


def load_knowledge() -> Tuple[List[str], np.ndarray, List[Dict]]:
    """
    Loads SAP TCode knowledge base and returns:
        texts      -> list[str]
        embeddings -> np.ndarray
        metadata   -> list[dict]
    """

    print(f"\n📂 Loading knowledge base from: {DATA_DIR}\n")

    # -------------------------
    # Load TCode dataset
    # -------------------------
    tcodes_file = DATA_DIR / "tcodes.csv"
    df = _safe_read_csv(tcodes_file)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required_columns = {"tcode", "description"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"\n❌ tcodes.csv must contain columns: {required_columns}\n"
            f"Found columns: {list(df.columns)}"
        )

    # -------------------------
    # Prepare knowledge records
    # -------------------------
    texts: List[str] = []
    metadata: List[Dict] = []

    for _, row in df.iterrows():
        tcode = str(row["tcode"]).strip()
        description = str(row["description"]).strip()

        text = f"{tcode} - {description}"

        texts.append(text)
        metadata.append(
            {
                "tcode": tcode,
                "description": description,
                "source": "tcodes.csv",
            }
        )

    # ------------------------------------
    # Load embeddings if precomputed exist
    # ------------------------------------
    embeddings_file = DATA_DIR / "embeddings.npy"

    if embeddings_file.exists():
        print("⚡ Loading precomputed embeddings")
        embeddings = np.load(embeddings_file)
    else:
        print("⚠ No embeddings.npy found — embeddings will be generated at runtime")
        embeddings = None

    print(f"✅ Loaded {len(texts)} tcodes\n")

    return texts, embeddings, metadata