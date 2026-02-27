# app/services/knowledge_loader.py
"""
Knowledge loader for SAP T-codes.

Responsibilities:
- Load structured T-code CSV from configured data folder (default: /app/data/tcodes.csv)
- Build the *texts* to embed from DESCRIPTION (+ optional MODULE)
- Load cached embeddings (embeddings.npy) when present and valid
- Otherwise compute embeddings using embed_query() from embedding_service
- Return (embeddings_matrix, metadata_list)

Return contract:
    load_knowledge() -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]
Where:
    embeddings_matrix : np.ndarray shape (N, D)  OR None on failure
    metadata_list     : list[dict] length N (each dict contains tcode, description, module) OR None on failure

Notes:
- This module is defensive: it prints helpful messages and never raises for expected runtime conditions.
- It tries to cache embeddings to embeddings.npy inside the same data directory so subsequent cold starts are fast.
"""

from typing import List, Dict, Optional, Tuple
import os
import time

import numpy as np
import pandas as pd

from app.services.embedding_service import embed_query  # single-item embed function

# Configuration
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
TCODES_FILE = os.environ.get("TCODES_FILE", os.path.join(DATA_DIR, "tcodes.csv"))
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """Read CSV defensively and return DataFrame or None."""
    try:
        if not os.path.exists(path):
            print(f"Render path checked: {path}")
            print(f"Make sure the file exists inside the {DATA_DIR} folder in your repo.")
            return None
        df = pd.read_csv(path)
        return df
    except Exception as exc:
        print(f"Failed to read CSV {path}: {exc}")
        return None


def _build_texts(df: pd.DataFrame) -> List[str]:
    """
    Build the text strings that will be embedded.
    Use `description` plus `module` (if available). Exclude the tcode itself from embedding.
    """
    desc = df["description"].astype(str).fillna("")
    if "module" in df.columns:
        module = df["module"].astype(str).fillna("")
        # join in a natural way so embeddings capture module context
        texts = (desc + " in SAP " + module).tolist()
    else:
        texts = desc.tolist()
    return texts


def load_knowledge() -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
    """
    Load the tcode knowledge and return (embeddings_matrix, metadata_list).

    - If precomputed embeddings.npy exists and matches the CSV row count, load and return it.
    - Otherwise generate embeddings by calling embed_query(text) for each row,
      save embeddings.npy for future cold starts, and return the computed matrix.

    On any fatal error return (None, None) so caller can handle warming / retry logic.
    """
    try:
        print(f"📂 Loading knowledge base from: {DATA_DIR}")

        df = _safe_read_csv(TCODES_FILE)
        if df is None:
            print(f"❌ DATA FILE MISSING: {TCODES_FILE}")
            return None, None

        # Required column check
        if "tcode" not in df.columns or "description" not in df.columns:
            print("❌ CSV missing required columns. Expected at least: tcode, description, optional: module")
            print(f"Found columns: {list(df.columns)}")
            return None, None

        # Normalize dataframe: ensure strings
        df["tcode"] = df["tcode"].astype(str).fillna("")
        df["description"] = df["description"].astype(str).fillna("")
        if "module" in df.columns:
            df["module"] = df["module"].astype(str).fillna("")

        # Build metadata structure
        metadata: List[Dict] = []
        for _, row in df.iterrows():
            metadata.append(
                {
                    "tcode": row.get("tcode", ""),
                    "description": row.get("description", ""),
                    "module": row.get("module", "") if "module" in df.columns else "",
                }
            )

        n_rows = len(metadata)
        print(f"✅ Loaded {n_rows} tcodes")

        # Try to load cached embeddings
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                arr = np.load(EMBEDDINGS_FILE)
                if arr is not None and getattr(arr, "shape", None) is not None and arr.shape[0] == n_rows:
                    print("🔁 Found embeddings.npy and row count matches — using cached embeddings")
                    return arr.astype(np.float32), metadata
                else:
                    print("⚠️ embeddings.npy found but shape mismatch — regenerating embeddings")
            except Exception as exc:
                print("⚠️ Failed to load embeddings.npy, regenerating:", str(exc))

        # No cached embeddings — generate at runtime
        print("⚠️ No embeddings.npy found — embeddings will be generated at runtime")
        texts = _build_texts(df)

        embeddings_list: List[List[float]] = []
        start = time.time()

        for i, text in enumerate(texts):
            try:
                emb = embed_query(text)
                if emb is None:
                    raise RuntimeError(f"embed_query returned None for row {i} text={text[:40]!r}")
                # ensure it's a 1-D numeric sequence
                emb_arr = np.array(emb, dtype=np.float32).reshape(-1)
                embeddings_list.append(emb_arr.tolist())
            except Exception as exc:
                # Log the exact failing text to help debugging (do NOT print full secrets)
                print(f"❌ Failed to embed row {i} ({metadata[i].get('tcode')}): {exc}")
                # Return failure so caller can handle warming_up or raise as needed
                return None, None

        # Convert to numpy array
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        # Persist for faster cold starts next time (best-effort)
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            np.save(EMBEDDINGS_FILE, embeddings_matrix)
            print(f"💾 Saved embeddings cache to {EMBEDDINGS_FILE}")
        except Exception as exc:
            print("⚠️ Failed to write embeddings.npy cache (continuing):", exc)

        duration = time.time() - start
        print(f"🔢 Generated {n_rows} embeddings in {duration:.2f}s, dim={embeddings_matrix.shape[1]}")

        return embeddings_matrix, metadata

    except Exception as exc:
        # Catch-all: print traceback and return None to signal failure to caller
        import traceback as _tb

        print("❌ Unexpected error in load_knowledge():", str(exc))
        _tb.print_exc()
        return None, None