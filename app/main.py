from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# =========================================================
# CONFIG
# =========================================================
FRONTEND_ORIGINS = [
    "https://sap-tcode-frontend.vercel.app",   # replace this
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]


# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="SAP T-Code Assistant API",
    version="1.0.0",
    description="Chat API for SAP T-code assistant frontend and integrations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================
class ChatRequest(BaseModel):
    query: str


class SearchRequest(BaseModel):
    query: str


# =========================================================
# SEARCH SERVICE ADAPTER
# Tries multiple possible function names from your existing
# app/services/search_service.py so you do not have to refactor
# the whole backend immediately.
# =========================================================
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
    """
    Normalizes many possible service return formats into one stable API shape.
    """

    if raw is None:
        return {
            "query": query,
            "answer": "No result found.",
            "speech": "I could not find a matching SAP T-code for that request.",
            "confidence": None,
            "confidence_label": "low",
            "best_match": None,
            "results": [],
        }

    # If service already returns a dict, adapt it
    if isinstance(raw, dict):
        speech = raw.get("speech")
        answer = (
            raw.get("answer")
            or raw.get("message")
            or raw.get("response")
            or speech
        )

        best_match = raw.get("best_match")
        results = raw.get("results", [])

        # Try to derive best_match from results if missing
        if not best_match and isinstance(results, list) and results:
            best_match = results[0]

        # If no answer but there is a best match, build one
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
            "confidence": raw.get("confidence"),
            "confidence_label": raw.get("confidence_label"),
            "best_match": best_match,
            "results": results if isinstance(results, list) else [],
            "type": raw.get("type"),
        }

    # If service returns a list, assume list of matches
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
            "confidence": None,
            "confidence_label": None,
            "best_match": best_match,
            "results": raw,
            "type": "search_results",
        }

    # Fallback for string/object output
    text = str(raw)
    return {
        "query": query,
        "answer": text,
        "speech": text,
        "confidence": None,
        "confidence_label": None,
        "best_match": None,
        "results": [],
        "type": "text",
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


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def root() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "SAP T-Code Assistant backend is running."
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


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


# =========================================================
# LOCAL RUN
# Render will ignore this and use its own start command.
# =========================================================
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)