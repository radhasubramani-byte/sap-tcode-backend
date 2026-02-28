# app/voice_webhook.py
import os
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, Header, HTTPException
import httpx
import asyncio

logger = logging.getLogger("voice_webhook")
router = APIRouter()

# Environment-configurable fallback search URL if internal import isn't available
LOCAL_SEARCH_FALLBACK_URL = os.getenv("LOCAL_SEARCH_FALLBACK_URL", "http://127.0.0.1:10000/search-tcode")
# The environment variable you should set in Render for webhook verification
WEBHOOK_SECRET_ENV = os.getenv("WEBHOOK_SECRET", None)

# Try to import the internal search function if available. We support multiple names for robustness.
_search_fn = None
try:
    from app.services import search_service  # type: ignore
    # prefer canonical names if available
    for name in ("search_tcode", "search", "semantic_search", "query"):
        _search_fn = getattr(search_service, name, None)
        if callable(_search_fn):
            logger.info("voice_webhook: bound internal search function: %s", name)
            break
except Exception:
    _search_fn = None


def verify_webhook_secret(header_secret: Optional[str], authorization: Optional[str]) -> bool:
    """
    Accept either:
      - X-Webhook-Secret: <token>
      - Authorization: Bearer <token>
    The token must match WEBHOOK_SECRET env var if set. If WEBHOOK_SECRET isn't set, allow (but log).
    """
    configured = WEBHOOK_SECRET_ENV
    token = None
    if header_secret:
        token = header_secret.strip()
    elif authorization:
        # header like 'Bearer <token>'
        if authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
        else:
            token = authorization.strip()

    if configured:
        ok = (token == configured)
        if not ok:
            logger.warning("voice_webhook: invalid webhook secret provided")
        return ok
    else:
        # no secret configured — allow but warn
        logger.warning("voice_webhook: WEBHOOK_SECRET not set in env; accepting request without verification")
        return True


async def call_local_search_via_http(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Fallback: call local HTTP search endpoint (assumes JSON response)."""
    payload = {"q": query, "top_k": top_k}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(LOCAL_SEARCH_FALLBACK_URL, params=payload)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.exception("voice_webhook: local HTTP search call failed: %s", e)
        return {"status": "error", "error": str(e)}


def map_confidence_label(score: float) -> str:
    """
    Returns confidence label based on score:
      - high: >= 0.75
      - medium: >= 0.4
      - low: otherwise
    """
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def build_speech_for_confidence(label: str, best_match: Dict[str, Any], alternatives: Optional[list]) -> str:
    """
    Generate natural spoken text that sounds like an 'AI consultant' responding differently
    depending on confidence.
    """
    desc = best_match.get("description") or best_match.get("desc") or ""
    tcode = best_match.get("tcode") or best_match.get("code") or ""
    if label == "high":
        return f"I'm pretty confident this is {tcode} — {desc}. Would you like me to open it or show the steps?"
    if label == "medium":
        alt_str = ""
        if alternatives:
            # give 1-2 short alternatives
            picks = [a.get("tcode") for a in alternatives[:2] if a.get("tcode")]
            if picks:
                alt_str = " Alternatives could be " + ", ".join(picks) + "."
        return f"I think the best match is {tcode} — {desc}.{alt_str} Do you mean this one?"
    # low confidence
    examples = "for example, 'create purchase requisition' or 'post vendor invoice'"
    return f"I need a little more detail to be sure which transaction you mean ({examples}). Can you rephrase or give an example?"

async def run_search_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Unified call to your search: try internal function first, otherwise HTTP fallback.
    Expected canonical response shape (best effort):
      {
        "status": "ok",
        "query": "...",
        "results": [...],
        "best_match": {...}
      }
    If internal function returns a plain list or other format, this tries to normalise.
    """
    # 1) internal function
    if callable(_search_fn):
        try:
            # support sync or async search function
            result = _search_fn(query, top_k=top_k) if _search_fn.__code__.co_argcount >= 1 else _search_fn(query)
            if asyncio.iscoroutine(result):
                result = await result
            # if search function already returns canonical dict, pass through; otherwise try to normalise:
            if isinstance(result, dict):
                return result
            # if result is list of rows
            if isinstance(result, (list, tuple)):
                return {"status": "ok", "query": query, "results": result, "best_match": (result[0] if result else None)}
            # fallthrough
            return {"status": "ok", "query": query, "results": result, "best_match": None}
        except Exception as e:
            logger.exception("voice_webhook: internal search_fn failed: %s", e)
            # fall back to HTTP
    # 2) HTTP fallback
    return await call_local_search_via_http(query, top_k=top_k)


@router.post("/voice-webhook")
async def voice_webhook(request: Request, x_webhook_secret: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    """
    POST /voice-webhook
    Body JSON expected:
      {
        "session_id": "...",
        "transcript": "user spoken text",
        "metadata": { ... }   # optional
      }

    Returns:
      {
        "speech": "...",            # text to speak
        "type": "answer"|"clarification",
        "confidence": 0.82,
        "confidence_label": "high",
        "results": [...],           # original search results (optional)
        "best_match": {...}         # the best result
      }
    """
    # verify secret
    if not verify_webhook_secret(x_webhook_secret, authorization):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload: Dict[str, Any] = await request.json()
    session_id = payload.get("session_id")
    transcript = (payload.get("transcript") or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Missing transcript")

    # perform search
    search_resp = await run_search_query(transcript, top_k=5)

    # attempt to extract best_match + score
    best = search_resp.get("best_match") or None
    results = search_resp.get("results") or []
    # many legacy responses put best match as results[0]
    if not best and results:
        best = results[0]

    # try to find a numeric score on best
    score = None
    if isinstance(best, dict):
        for k in ("score", "confidence", "sim"):
            v = best.get(k)
            if isinstance(v, (int, float)):
                score = float(v)
                break
            # sometimes it's a string numeric
            if isinstance(v, str):
                try:
                    score = float(v)
                    break
                except Exception:
                    pass

    # default score if missing
    if score is None:
        # use a conservative default
        score = 0.0

    conf_label = map_confidence_label(score)
    speech = build_speech_for_confidence(conf_label, best or {}, results[1:] if len(results) > 1 else [])

    response = {
        "speech": speech,
        "type": "answer" if conf_label in ("high", "medium") else "clarification",
        "confidence": round(score, 3),
        "confidence_label": conf_label,
        "results": results,
        "best_match": best,
    }

    logger.info("voice_webhook: session=%s query=%s conf=%s label=%s", session_id, transcript, response["confidence"], conf_label)
    return response