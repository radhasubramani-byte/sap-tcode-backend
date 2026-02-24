from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

vector_store = None

def build_vector_store(documents):
    global vector_store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_texts(documents, embeddings)

def search_documents(query: str, k: int = 5):
    global vector_store

    if vector_store is None:
        return ["Knowledge not loaded"]

    docs = vector_store.similarity_search(query, k=k)
    return [d.page_content for d in docs]