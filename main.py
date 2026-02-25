import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# IMPORTANT: correct function name
from app.services.search_service import search_tcode

app = FastAPI(title="SAP TCode Semantic Search API")

# Allow frontend / tools to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "SAP TCode backend running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/search")
def search(q: str = Query(..., description="Natural language SAP request")):
    try:
        results = search_tcode(q)
        return {"query": q, "results": results}
    except Exception as e:
        return {"error": str(e)}


# Render requires binding to PORT env variable
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
