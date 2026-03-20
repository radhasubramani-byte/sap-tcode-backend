from typing import Any, Callable, Dict
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.services.search_service import initialize_search, is_ready


FRONTEND_ORIGINS = [
    "https://sap-tcode-frontend.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]


app = FastAPI(
    title="SAP T-Code Assistant API",
    version="1.0.0",
    description="Chat API for SAP T-code assistant frontend and integrations.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


class SearchRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup_event() -> None:
    thread = threading.Thread(target=initialize_search, daemon=True)
    thread.start()
    print("Background search initialization started")


def _load_search_callable() -> Callable[[str], Any]:
    try:
        import app.services.search_service as search_service_module
    except Exception as exc:
        raise RuntimeError(
            "Could not import app.services.search_service. "
            "Make sure your search service file exists."
        ) from exc

    candidate_names = [
        "search_tcode",
        "search",
        "find_tcode",
        "find_best_match",
        "query_tcode",
        "run_search",
    ]

    for name in candidate_names:
        fn = getattr(search_service_module, name, None)
        if callable(fn):
            return fn

    raise RuntimeError(
        "No usable search function found in app.services.search_service. "
        "Expected one of: search_tcode, search, find_tcode, "
        "find_best_match, query_tcode, run_search"
    )


def _normalize_result(query: str, raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {
            "query": query,
            "answer": "No result found.",
            "speech": "I could not find a matching SAP T-code for that request.",
            "best_match": None,
            "results": [],
            "type": "none",
            "ready": is_ready(),
        }

    if isinstance(raw, dict):
        speech = raw.get("speech")
        answer = raw.get("answer") or raw.get("message") or raw.get("response") or speech

        best_match = raw.get("best_match")
        results = raw.get("results", [])

        if not best_match and isinstance(results, list) and results:
            best_match = results[0]

        if not answer and isinstance(best_match, dict):
            code = best_match.get("tcode") or best_match.get("code") or best_match.get("t_code")
            desc = best_match.get("description") or best_match.get("task") or best_match.get("text")
            module = best_match.get("module")

            if code and desc and module:
                answer = f"The recommended SAP T-code is {code} for {desc} in module {module}."
            elif code and desc:
                answer = f"The recommended SAP T-code is {code} for {desc}."
            elif code:
                answer = f"The recommended SAP T-code is {code}."

        if not speech:
            speech = answer or "No result found."

        return {
            "query": query,
            "answer": answer or "No result found.",
            "speech": speech,
            "best_match": best_match,
            "results": results if isinstance(results, list) else [],
            "type": raw.get("type"),
            "status": raw.get("status"),
            "ready": is_ready(),
        }

    if isinstance(raw, list):
        best_match = raw[0] if raw else None
        answer = "No result found."

        if isinstance(best_match, dict):
            code = best_match.get("tcode") or best_match.get("code") or best_match.get("t_code")
            desc = best_match.get("description") or best_match.get("task") or best_match.get("text")
            module = best_match.get("module")

            if code and desc and module:
                answer = f"The recommended SAP T-code is {code} for {desc} in module {module}."
            elif code and desc:
                answer = f"The recommended SAP T-code is {code} for {desc}."
            elif code:
                answer = f"The recommended SAP T-code is {code}."

        return {
            "query": query,
            "answer": answer,
            "speech": answer,
            "best_match": best_match,
            "results": raw,
            "type": "search_results",
            "ready": is_ready(),
        }

    text = str(raw)
    return {
        "query": query,
        "answer": text,
        "speech": text,
        "best_match": None,
        "results": [],
        "type": "text",
        "ready": is_ready(),
    }


def run_search(query: str) -> Dict[str, Any]:
    try:
        search_fn = _load_search_callable()
        raw = search_fn(query)
        return _normalize_result(query, raw)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Search service execution failed: {exc}") from exc


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": "SAP T-Code Assistant backend is running.",
        "search_ready": is_ready(),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "search_ready": is_ready(),
    }


@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        return run_search(query)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search-tcode")
def search_tcode(request: SearchRequest) -> Dict[str, Any]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        return run_search(query)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)