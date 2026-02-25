from fastapi import FastAPI
from fastapi.responses import JSONResponse
import threading
import time

app = FastAPI()

# -------------------------------------------
# HEALTH CHECK (Render requirement)
# -------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# -------------------------------------------
# BACKGROUND INITIALIZER
# -------------------------------------------
def start_initializer():
    from app.services.search_service import initialize_search
    print("Starting background knowledge initialization...")
    initialize_search()


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=start_initializer, daemon=True)
    thread.start()


# -------------------------------------------
# SEARCH ENDPOINT
# -------------------------------------------
@app.get("/search")
def search(q: str):
    from app.services.search_service import search_tcode, is_ready

    if not is_ready():
        return JSONResponse({
            "status": "warming_up",
            "message": "AI knowledge base loading (~20s first deploy)"
        })

    results = search_tcode(q)
    return {"results": results}
