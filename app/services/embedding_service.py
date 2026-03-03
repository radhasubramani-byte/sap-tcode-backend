# app/services/embedding_service.py
import os
from typing import List, Union, Optional

import numpy as np

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Cached client (lazy-init)
_client = None


def _get_client():
    """
    Create OpenAI client only when needed.
    This prevents import-time crashes when OPENAI_API_KEY is not set (local dev).
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment before calling embed_query()."
        )
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install dependencies.")

    _client = OpenAI(api_key=api_key)
    return _client


def embed_query(text: Union[str, List[str]], model: str = "text-embedding-3-large"):
    """
    Accepts a single string or list of strings.
    Returns:
      - np.ndarray (D,) for single string
      - np.ndarray (N, D) for list[str]
    """
    client = _get_client()

    # Normalize input to list
    is_single = isinstance(text, str)
    inputs = [text] if is_single else list(text)

    resp = client.embeddings.create(model=model, input=inputs)
    vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    if is_single:
        return vectors[0]
    return np.vstack(vectors)