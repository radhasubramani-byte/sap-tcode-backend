from __future__ import annotations

import os
import traceback
from typing import Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

try:
    from app.services.search_service import search_tcode, is_ready as search_is_ready
except Exception:
    search_tcode = None  # type: ignore
    search_is_ready = lambda: False  # type: ignore

router = APIRouter(prefix="/voice-webhook", tags=["voice"])


# -------------------------
# Models
# -------------------------

class VoiceWebhookRequest(BaseModel):
    session_id: str = Field(default="unknown")
    transcript: str = Field(default="")
    message: str = Field(default="")
    query: str = Field(default="")


class VoiceWebhookResponse(BaseModel):
    speech: str
    type: str = Field(default="error")  # confident | uncertain | none | error | warming_up
    confidence: float = Field(default=0.0)
    confidence_label: str = Field(default="low")
    best_match: Optional[Dict] = None
    results: List[Dict] = Field(default_factory=list)


# -------------------------
# TTS helpers
# -------------------------

_DIGITS = "zero one two three four five six seven eight nine".split()


def speak_tcode(code: str) -> str:
    """
    Speak SAP T-codes clearly for phone TTS.
    Examples:
      ME21N -> M E two one N
      VF03  -> V F zero three
      /SAPPO/PPO3 -> slash S A P P O slash P P O three
    """
    code = (code or "").strip().upper()
    out: List[str] = []

    for c in code:
        if c.isdigit():
            out.append(_DIGITS[int(c)])
        elif c == "/":
            out.append("slash")
        elif c == "-":
            out.append("dash")
        elif c == "_":
            out.append("underscore")
        else:
            out.append(c)

    return " ".join(out)


def speak_letters(s: str) -> str:
    s = (s or "").strip().upper()
    if not s:
        return ""
    return " ".join(list(s))


def sap_prefix() -> str:
    # No punctuation between SAP and T-code; sounds better in VAPI/TTS
    return "S A P T-code"


def naturalize_description(desc: str) -> str:
    """
    Make SAP descriptions sound more natural on phone calls.
    """
    desc = (desc or "").strip()

    if not desc:
        return ""

    d = desc.lower()

    replacements = {
        "create purchase order": "create a purchase order",
        "change purchase order": "change a purchase order",
        "display purchase order": "display a purchase order",
        "create purchase requisition": "create a purchase requisition",
        "change purchase requisition": "change a purchase requisition",
        "display purchase requisition": "display a purchase requisition",
        "display billing document": "display a billing document",
        "create billing document": "create a billing document",
        "goods movement": "post goods movement",
        "goods receipt": "post goods receipt",
        "goods issue": "post goods issue",
        "display customer invoice": "display a customer invoice",
        "create sales order": "create a sales order",
        "change sales order": "change a sales order",
        "display sales order": "display a sales order",
    }

    if d in replacements:
        return replacements[d]

    # generic improvements
    if d.startswith("create "):
        return "create a " + d[len("create "):]
    if d.startswith("change "):
        return "change a " + d[len("change "):]
    if d.startswith("display "):
        return "display a " + d[len("display "):]

    return d


def dedupe_results(results: List[Dict]) -> List[Dict]:
    """
    Remove duplicate T-codes from results while keeping order.
    Helpful when same alias exists in multiple alias CSV files.
    """
    seen = set()
    out: List[Dict] = []

    for r in results:
        if not isinstance(r, dict):
            continue

        tcode = (r.get("tcode") or "").strip().upper()
        key = tcode or f"{r.get('description')}|{r.get('module')}"

        if key in seen:
            continue

        seen.add(key)
        out.append(
            {
                "tcode": r.get("tcode"),
                "description": r.get("description"),
                "module": r.get("module"),
                "score": r.get("score"),
                "match_type": r.get("match_type"),
            }
        )

    return out


def build_speech(best_match: Optional[Dict], results: List[Dict], c_label: str, r_type: str) -> str:
    """
    Voice-optimized speech builder.
    Keeps SAP pronunciation clear and makes the sentence sound natural on phone calls.
    """
    results = dedupe_results(results)

    if not best_match:
        if results:
            options = []
            for r in results[:3]:
                tc = (r.get("tcode") or "").strip()
                dd = naturalize_description(r.get("description") or "")
                mod = (r.get("module") or "").strip()
                mod_sp = speak_letters(mod)

                if tc and dd:
                    line = f"{speak_tcode(tc)} to {dd}"
                elif tc:
                    line = speak_tcode(tc)
                else:
                    line = dd

                if mod_sp:
                    line = f"{line} in module {mod_sp}"

                options.append(line)

            joined = " or ".join(options)
            return f"I found a few close options: {joined}. Please choose one."

        return (
            "I couldn’t find a matching S A P T-code. "
            "Try saying something like create a purchase order, create a purchase requisition, "
            "post goods receipt, or display a profit center."
        )

    tcode = (best_match.get("tcode") or "").strip()
    desc = naturalize_description(best_match.get("description") or "")
    module = (best_match.get("module") or "").strip()

    prefix = sap_prefix()
    spoken_code = speak_tcode(tcode)
    module_spoken = speak_letters(module)

    if r_type == "uncertain":
        opts = []
        for r in results[:3]:
            tc = (r.get("tcode") or "").strip()
            dd = naturalize_description(r.get("description") or "")
            mod = (r.get("module") or "").strip()
            mod_sp = speak_letters(mod)

            if tc and dd:
                line = f"{speak_tcode(tc)} to {dd}"
            elif tc:
                line = speak_tcode(tc)
            else:
                line = dd

            if mod_sp:
                line = f"{line} in module {mod_sp}"

            opts.append(line)

        options = " or ".join(opts) if opts else f"{spoken_code} to {desc}"
        return f"I found a few close options: {options}. Please choose one."

    # Confident / normal case
    if desc and module_spoken:
        return f"The recommended {prefix} is {spoken_code} to {desc} in module {module_spoken}."
    if desc:
        return f"The recommended {prefix} is {spoken_code} to {desc}."
    return f"The recommended {prefix} is {spoken_code}."


# -------------------------
# Helpers
# -------------------------

def _check_bearer_token(authorization: Optional[str]) -> None:
    expected = (
        os.getenv("WEBHOOK_SECRET")
        or os.getenv("VOICE_WEBHOOK_TOKEN")
        or os.getenv("SAP_VOICE_WEBHOOK_TOKEN")
    )

    if not expected:
        return

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    got = authorization.split(" ", 1)[1].strip()
    if got != expected:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")


def _extract_transcript(payload: VoiceWebhookRequest) -> str:
    return (payload.transcript or payload.message or payload.query or "").strip()


# -------------------------
# Main endpoint
# -------------------------

@router.post("", response_model=VoiceWebhookResponse)
def voice_webhook(
    payload: VoiceWebhookRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> VoiceWebhookResponse:
    _check_bearer_token(authorization)

    transcript = _extract_transcript(payload)
    print(f"VOICE transcript: {transcript}")

    if not transcript:
        return VoiceWebhookResponse(
            speech="Please say what you want to do in S A P, for example: create a purchase order.",
            type="none",
            confidence=0.0,
            confidence_label="low",
            best_match=None,
            results=[],
        )

    if not callable(search_tcode):
        return VoiceWebhookResponse(
            speech="Sorry, the search service is not available right now.",
            type="error",
            confidence=0.0,
            confidence_label="low",
            best_match=None,
            results=[],
        )

    try:
        # Keep warm-up message only when backend truly cannot search yet.
        if not search_is_ready():
            print("VOICE readiness check: search service reports not ready yet")
    except Exception as exc:
        print(f"VOICE readiness check failed: {exc}")

    try:
        # Voice should follow the exact same ranking as chat/search service.
        search_resp = search_tcode(transcript, top_k=5)
        print(f"VOICE search response: {search_resp}")

        if not isinstance(search_resp, dict):
            search_resp = {"message": str(search_resp)}

        if search_resp.get("status") == "warming_up":
            return VoiceWebhookResponse(
                speech="One moment — I’m loading the S A P knowledge base.",
                type="warming_up",
                confidence=0.0,
                confidence_label="low",
                best_match=None,
                results=[],
            )

        best = search_resp.get("best_match")
        results = search_resp.get("results") or []

        if not isinstance(results, list):
            results = []

        results = dedupe_results(results)

        if not best and results and isinstance(results[0], dict):
            best = results[0]

        score_raw = search_resp.get("confidence", 0.0)
        try:
            score = float(score_raw or 0.0)
        except Exception:
            score = 0.0

        c_label = search_resp.get("confidence_label") or (
            "high" if score >= 0.70 else "medium" if score >= 0.40 else "low"
        )

        r_type = search_resp.get("type")
        if r_type not in {"confident", "uncertain", "none", "warming_up", "error"}:
            r_type = "confident" if best else "none"

        # Prefer backend text only for explicit uncertain / error style messages.
        backend_speech = (
            search_resp.get("speech")
            or search_resp.get("answer")
            or search_resp.get("message")
        )

        if r_type == "uncertain" and not best:
            if backend_speech:
                return VoiceWebhookResponse(
                    speech=str(backend_speech),
                    type="uncertain",
                    confidence=round(score, 3),
                    confidence_label=c_label,
                    best_match=None,
                    results=results,
                )

            speech = build_speech(None, results, c_label, "uncertain")
            return VoiceWebhookResponse(
                speech=speech,
                type="uncertain",
                confidence=round(score, 3),
                confidence_label=c_label,
                best_match=None,
                results=results,
            )

        if best and isinstance(best, dict):
            speech = build_speech(best, results, c_label, r_type)
            return VoiceWebhookResponse(
                speech=speech,
                type="confident" if r_type == "confident" else r_type,
                confidence=round(score, 3),
                confidence_label=c_label,
                best_match=best,
                results=results,
            )

        if backend_speech:
            return VoiceWebhookResponse(
                speech=str(backend_speech),
                type=r_type,
                confidence=round(score, 3),
                confidence_label=c_label,
                best_match=None,
                results=results,
            )

        speech = build_speech(None, results, c_label, r_type)
        return VoiceWebhookResponse(
            speech=speech,
            type=r_type,
            confidence=round(score, 3),
            confidence_label=c_label,
            best_match=None,
            results=results,
        )

    except Exception as exc:
        print("voice_webhook error:", repr(exc))
        print(traceback.format_exc())
        return VoiceWebhookResponse(
            speech="Sorry, I’m having trouble reaching the S A P assistant right now. Please try again.",
            type="error",
            confidence=0.0,
            confidence_label="low",
            best_match=None,
            results=[],
        )


@router.get("/health")
def voice_webhook_health():
    return {
        "search_ready": bool(search_is_ready()) if callable(search_is_ready) else False,
        "webhook_registered": True,
        "route": "/voice-webhook",
    }
