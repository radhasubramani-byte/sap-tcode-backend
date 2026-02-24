import os
from app.services.knowledge_loader import load_knowledge

print("Building SAP Knowledge Embeddings...")

load_knowledge(persist=True)

print("Embeddings built successfully")