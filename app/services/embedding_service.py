# app/services/embedding_service.py

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "text-embedding-3-small"


def embed_query(text: str):
    """Create embedding for a search query"""
    response = client.embeddings.create(
        model=MODEL,
        input=text
    )
    return response.data[0].embedding


def embed_documents(texts):
    """Create embeddings for many texts"""
    response = client.embeddings.create(
        model=MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]