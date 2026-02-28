# app/services/voice_webhook.py
import os
import re
import threading
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import JSONResponse

# Try to import the search service module and bind available search function
import importlib

_log = logging.getLogger("voice_webhook")
_log.setLevel(logging.INFO)

router = APIRouter()

# simple in-memory pending actions per session (session_id -> action dict)
_pending_lock = threading.Lock()
_pending_actions: Dict[str, Dict[str, Any]] = {}

# load search_service module and bind candidate names
svc_mod = importlib.import_module("app.services.search_service")
_search_candidates = ["search_tcode", "search", "semantic_search", "query", "_search"]
search_fn = None
for name in _search_candidates:
    fn = getattr(svc_mod, name, None)
    if callable(fn):
        search_fn = fn
        _bound_name = name
        break
else:
    _bound_name = None

# optional secret: set WEBHOOK_SECRET in Render env and configure VAPI to send Authorization: Bearer <secret>
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")


def _verify_secret(auth_header: Optional[str]):
    if not WEBHOOK_SECRET:
        return True
    if not auth_header:
        return False
    # support "Bearer <token>"
    m = re.match(r"Bearer\s+(.+)", auth_header or "", flags=re.I)
    if not m:
        return False
    token = m.group(1).strip()
    return token == WEBHOOK_SECRET


def _is_affirmative(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.match(r"^(yes|yeah|yup|sure|please do|confirm|go ahead|do it|ok|okay)\b", t))


def _is_negative(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.match(r"^(no|nope|nah|don't|do not|stop|cancel)\b", t))


def _format_ssml(speech: str) -> str:
    # small convenience: wrap in speak tag
    return f"<speak>{speech}</speak>"


def _choose_reply_from_search(search_json: Dict[str, Any]) -> (str, Dict[str, Any]):
    """
    Return (speech_text, action)
    action is a dict containing: tcode (or None), confidence (0..1), level ("high"/"medium"/"low")
    """
    if not search_json:
        return ("I couldn't reach the SAP search service. Please try again shortly.", {"tcode": None, "confidence": 0.0, "level": "low"})

    # pass through "warming_up" and error states
    if isinstance(search_json, dict) and search_json.get("status") in ("warming_up", "invalid_query", "error"):
        if search_json.get("status") == "warming_up":
            return ("The SAP assistant is still starting up — I can try again in a few seconds.", {"tcode": None, "confidence": 0.0, "level": "low"})
        return (search_json.get("message", "An error occurred"), {"tcode": None, "confidence": 0.0, "level": "low"})

    # Expected canonical response:
    # { "status":"ok", "results":[...], "best_match": {...}, "confidence":0.80, "confidence_label":"high" }
    conf = float(search_json.get("confidence") or 0.0)
    conf_label = (search_json.get("confidence_label") or "").lower()
    best = search_json.get("best_match") or {}
    tcode = best.get("tcode")
    desc = best.get("description") or ""

    # High confidence -> assertive, offer action
    if conf >= 0.75 or conf_label == "high":
        speech = f"I'm confident this is {tcode} — {desc}. Would you like me to open that transaction or walk you through the steps?"
        action = {"tcode": tcode, "confidence": conf, "level": "high"}
        return speech, action

    # Medium confidence -> provide alternative and ask for clarification
    if 0.40 <= conf < 0.75 or conf_label == "medium":
        # try to pick one alternative
        alternatives = search_json.get("results", [])
        alt = alternatives[1] if len(alternatives) > 1 else None
        if alt:
            alt_t = alt.get("tcode"); alt_d = alt.get("description", "")
            speech = (f"I think it's {tcode} — {desc}, but a close alternative is {alt_t} — {alt_d}. "
                      "Which one did you mean, the first or the alternative?")
        else:
            speech = f"I believe it's {tcode} — {desc}. Do you want me to proceed with that?"
        action = {"tcode": tcode, "confidence": conf, "level": "medium"}
        return speech, action

    # Low confidence -> ask clarifying question with examples
    examples = ["create purchase requisition", "post vendor invoice", "display purchase order"]
    examples_text = "; ".join(examples[:3])
    speech = ("I need a bit more detail to identify the correct SAP transaction. "
              f"For example, you can say: {examples_text}. What would you like to do?")
    action = {"tcode": None, "confidence": conf, "level": "low"}
    return speech, action


@router.post("/voice-webhook")
async def voice_webhook(request: Request, authorization: Optional[str] = Header(None)):
    """
    Expected JSON body:
    {
      "session_id": "<optional session id>",
      "transcript": "user spoken text",
      "last_action": { ... }   # optional, if VAPI keeps session client-side
    }

    Response:
    {
      "speech": "text to say",
      "ssml": "<speak>...</speak>",
      "action": { "tcode": "...", "confidence":0.8, "level":"high" }
    }
    """
    # auth check
    if not _verify_secret(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    transcript = (body.get("transcript") or "").strip()
    session_id = body.get("session_id")
    client_last_action = body.get("last_action")  # optional

    if not transcript:
        return JSONResponse({"speech": "I didn't catch that. Could you repeat?"})

    _log.info("voice-webhook received: session=%s transcript=%s", session_id, transcript[:120])

    # If user replied with a simple yes/no and we have a pending action, handle it
    if session_id:
        with _pending_lock:
            pending = _pending_actions.get(session_id)
    else:
        pending = None

    # if user confirms and pending exists
    if pending and _is_affirmative(transcript):
        tcode = pending.get("tcode")
        _log.info("User confirmed pending action for session %s -> %s", session_id, tcode)
        # clear pending action
        if session_id:
            with _pending_lock:
                _pending_actions.pop(session_id, None)
        # produce confirm speech (this is what agent will say next)
        speech = f"Okay — opening {tcode} now and I will walk you through the top fields."
        ssml = _format_ssml(speech)
        return JSONResponse({"speech": speech, "ssml": ssml, "action": {"tcode": tcode, "confirmed": True}})

    # if user declines
    if pending and _is_negative(transcript):
        if session_id:
            with _pending_lock:
                _pending_actions.pop(session_id, None)
        speech = "Okay, I won't proceed. How else can I help with SAP?"
        return JSONResponse({"speech": speech, "ssml": _format_ssml(speech), "action": {"tcode": None, "confirmed": False}})

    # Otherwise call the search service
    if not search_fn:
        _log.error("No search function bound in search_service module")
        return JSONResponse({"speech": "The backend search service is not available right now."}, status_code=500)

    # Call search function defensively. Support either (q, top_k) or (q) signature.
    try:
        try:
            search_resp = search_fn(transcript, top_k=5)
        except TypeError:
            search_resp = search_fn(transcript)
    except Exception as e:
        _log.exception("Search function raised")
        return JSONResponse({"speech": "Search failed; please try again."}, status_code=500)

    # Build reply and action
    speech_text, action = _choose_reply_from_search(search_resp)

    # If the action level is medium/high, keep it as pending for confirmation
    if session_id and action and action.get("tcode") and action.get("level") in ("medium", "high"):
        with _pending_lock:
            _pending_actions[session_id] = action

    ssml = _format_ssml(speech_text)
    _log.info("Responding speech (len=%d) session=%s", len(speech_text), session_id)
    return JSONResponse({"speech": speech_text, "ssml": ssml, "action": action})