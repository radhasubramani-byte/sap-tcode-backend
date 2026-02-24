import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

# Lazy import to avoid heavy startup during Render boot scan
search_service = None

def get_search_service():
    global search_service
    if search_service is None:
        from search_service import SearchService
        search_service = SearchService()
    return search_service

app = FastAPI(title="SAP TCode Backend", version="1.0.0")


@app.get("/")
def root():
    return {"status": "ok", "service": "sap-tcode-backend"}


@app.get("/health")
def health():
    # very lightweight — Render uses this to verify service
    return {"status": "healthy"}


@app.get("/search")
def search(q: str = Query(..., min_length=1)):
    try:
        svc = get_search_service()
        results = svc.search(q)
        return JSONResponse({"query": q, "results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
