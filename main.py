import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import router
from app.services.search_service import search_tcode

app = FastAPI(title="SAP TCode Search API")

# CORS (important for frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "running", "service": "sap-tcode-backend"}

@app.get("/search")
def search(query: str):
    result = search_tcode(query)
    return {"results": result}


# IMPORTANT — Render needs this
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))  # Render injects PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)