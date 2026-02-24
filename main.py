import os
from fastapi import FastAPI, Query, HTTPException

# IMPORTANT: absolute package imports for Render/Linux
from app.services.search_service import search

app = FastAPI(title="SAP TCode Semantic Search API")


@app.get("/")
def root():
    return {"message": "SAP TCode backend is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/search")
def search_endpoint(q: str = Query(..., description="Natural language SAP query")):
    try:
        results = search(q)
        return {
            "query": q,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Render requires binding to its provided PORT
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
