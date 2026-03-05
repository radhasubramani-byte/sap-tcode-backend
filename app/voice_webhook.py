# app/voice_webhook.py
"""
Voice webhook router for SAP T-code Assistant.

Fixes:
- TTS pronunciation: always says "S. A. P. T code" (not "sap"/"sapt"/"s-a-p")
- Spells T-codes character-by-character for clarity on phone demos
- Deterministic alias priority + longest contains-match fallback
- Confidence threshold logic
- Robust DATA_DIR detection (Render + local)
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

# Prefer calling search_service directly
try:
    from app.services.search_service import search_tcode, is_ready as search_is_ready
except Exception:
    search_tcode = None  # type: ignore
    search_is_ready = lambda: False  # type: ignore


router = APIRouter()


# -------------------------
# Models
# -------------------------

class VoiceWebhookRequest(BaseModel):
    session_id: str = Field(default="unknown")
    transcript: str = Field(default="")

class VoiceWebhookResponse(BaseModel):
    speech: str
    type: str = Field(default="error")  # confident | uncertain | none | error | warming_up
    confidence: float = Field(default=0.0)
    confidence_label: str = Field(default="low")
    best_match: Optional[Dict] = None
    results: List[Dict] = Field(default_factory=list)


# -------------------------
# Robust DATA_DIR detection
# -------------------------

def detect_data_dir() -> Optional[str]:
    candidates = [
        os.getenv("DATA_DIR"),
        "/app/data",
        "/app/app/data",
        "data",
        "app/data",
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None

DATA_DIR = detect_data_dir()


# -------------------------
# Alias loading + matching
# -------------------------

_ALIAS_FILES_PRIORITY = [
    ("mm_aliases.csv", "MM", 1),
    ("sd_aliases.csv", "SD", 2),
    ("le_aliases.csv", "LE", 3),
]

@dataclass(frozen=True)
class AliasRecord:
    alias: str
    tcode: str
    canonical_desc: str
    module: str
    priority: int

_alias_map: Dict[str, AliasRecord] = {}
_alias_loaded: bool = False
_alias_load_error: Optional[str] = None


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[\u2018\u2019\u201C\u201D]", "'", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_aliases_once() -> None:
    global _alias_loaded, _alias_load_error, _alias_map

    if _alias_loaded:
        return
    _alias_loaded = True
    _alias_map = {}
    _alias_load_error = None

    if not DATA_DIR:
        _alias_load_error = "DATA_DIR not found"
        print("❌ No valid DATA_DIR detected for alias loading.")
        return

    any_loaded = False

    for filename, module, priority in _ALIAS_FILES_PRIORITY:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    alias_raw = (row.get("alias") or "").strip()
                    tcode = (row.get("tcode") or "").strip()
                    canonical_desc = (row.get("canonical_desc") or "").strip()

                    if not alias_raw or not tcode:
                        continue

                    key = _normalize_text(alias_raw)
                    rec = AliasRecord(
                        alias=alias_raw,
                        tcode=tcode,
                        canonical_desc=canonical_desc,
                        module=module,
                        priority=priority,
                    )

                    prev = _alias_map.get(key)
                    # lower numeric priority wins (MM=1 is highest)
                    if prev is None or rec.priority < prev.priority:
                        _alias_map[key] = rec

            any_loaded = True
            print(f"✅ Loaded aliases from {filename}: {path}")

        except Exception as exc:
            _alias_load_error = f"Failed reading {filename}: {exc}"
            print(f"❌ Failed to load {filename}: {exc}")

    if not any_loaded:
        print(f"⚠️ No alias files found in DATA_DIR={DATA_DIR}")


def match_alias(transcript: str) -> Optional[Dict]:
    """
    Deterministic alias priority:
    1) exact normalized match
    2) contains match: choose (highest priority, then longest alias)
       so phrases like "how do I post goods receipt" still match "post goods receipt"
    """
    _load_aliases_once()
    user_key = _normalize_text(transcript)
    if not user_key:
        return None

    # 1) Exact match
    rec = _alias_map.get(user_key)
    if rec:
        return {
            "tcode": rec.tcode,
            "description": rec.canonical_desc or None,
            "module": rec.module,
            "score": 1.0,
            "match_type": "alias_exact",
            "alias": rec.alias,
        }

    # 2) Contains match (longest wins within priority)
    best: Optional[Tuple[int, int, AliasRecord]] = None
    # tuple: (priority, -len(alias), rec)  -> smaller priority first, longer alias first
    for k, r in _alias_map.items():
        if k and k in user_key:
            cand = (r.priority, -len(k), r)
            if best is None or cand < best:
                best = cand

    if not best:
        return None

    rec2 = best[2]
    return {
        "tcode": rec2.tcode,
        "description": rec2.canonical_desc or None,
        "module": rec2.module,
        "score": 0.95,
        "match_type": "alias_contains",
        "alias": rec2.alias,
    }


# -------------------------
# Confidence thresholds
# -------------------------

CONFIDENT_THRESHOLD = float(os.getenv("CONFIDENT_THRESHOLD", "0.70"))
MEDIUM_THRESHOLD = float(os.getenv("MEDIUM_THRESHOLD", "0.40"))

def label_confidence(score: float) -> str:
    if score >= CONFIDENT_THRESHOLD:
        return "high"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"

def response_type_for_score(score: float) -> str:
    return "confident" if score >= MEDIUM_THRESHOLD else "uncertain"


# -------------------------
# TTS helpers (IMPORTANT)
# -------------------------

_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

def speak_tcode_slow(tcode: str) -> str:
    """
    Converts: ME21N -> "M … E … two … one … N"
    Adds strong pauses so VAPI TTS won't mash it.
    """
    tcode = (tcode or "").strip().upper()
    if not tcode:
        return ""
    parts = []
    for ch in tcode:
        if ch.isdigit():
            parts.append(_DIGIT_WORDS.get(ch, ch))
        else:
            parts.append(ch)
    return " … ".join(parts)


def build_speech(best_match: Optional[Dict], results: List[Dict], c_label: str, r_type: str) -> str:
    """
    IMPORTANT: To stop TTS saying "SAPT / SFPT / SPT",
    always use: "S. A. P." + "T code" (two words).
    """
    if not best_match:
        return (
            "I couldn’t find a matching S. A. P. transaction code. "
            "Try saying a task like create purchase order, post goods receipt, or change material."
        )

    tcode = best_match.get("tcode") or ""
    desc = best_match.get("description") or ""
    module = best_match.get("module") or ""

    spoken_code = speak_tcode_slow(tcode)

    if r_type == "confident" and c_label in ("high", "medium"):
        if module:
            return f"The S. A. P. T code is… {spoken_code}. {desc}. Module {module}."
        return f"The S. A. P. T code is… {spoken_code}. {desc}."

    # Uncertain: give up to 3 options, still spelled out
    opts = []
    for r in results[:3]:
        tc = r.get("tcode") or ""
        dd = r.get("description") or ""
        if tc:
            opts.append(f"{speak_tcode_slow(tc)} — {dd}")
    options = " / ".join(opts) if opts else f"{spoken_code} — {desc}"

    return f"I’m not fully confident. The closest options are: {options}. Which one did you mean?"


# -------------------------
# Auth
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


# -------------------------
# Main endpoint
# -------------------------

@router.post("/voice-webhook", response_model=VoiceWebhookResponse)
def voice_webhook(
    payload: VoiceWebhookRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> VoiceWebhookResponse:

    _check_bearer_token(authorization)

    transcript = (payload.transcript or "").strip()
    if not transcript:
        return VoiceWebhookResponse(
            speech="Please say what you want to do in S. A. P., for example: create purchase order.",
            type="none",
            confidence=0.0,
            confidence_label="low",
            best_match=None,
            results=[],
        )

    # 1) Alias (exact/contains) first
    alias_hit = match_alias(transcript)
    if alias_hit:
        speech = build_speech(alias_hit, [alias_hit], "high", "confident")
        return VoiceWebhookResponse(
            speech=speech,
            type="confident",
            confidence=1.0,
            confidence_label="high",
            best_match=alias_hit,
            results=[alias_hit],
        )

    # 2) Semantic search fallback
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
        if not search_is_ready():
            return VoiceWebhookResponse(
                speech="One moment — I’m loading the S. A. P. knowledge base.",
                type="warming_up",
                confidence=0.0,
                confidence_label="low",
                best_match=None,
                results=[],
            )
    except Exception:
        pass

    try:
        search_resp = search_tcode(transcript, top_k=5)

        if isinstance(search_resp, dict) and search_resp.get("status") == "warming_up":
            return VoiceWebhookResponse(
                speech="One moment — I’m loading the S. A. P. knowledge base.",
                type="warming_up",
                confidence=0.0,
                confidence_label="low",
                best_match=None,
                results=[],
            )

        best = None
        results: List[Dict] = []
        score = 0.0

        if isinstance(search_resp, dict):
            best = search_resp.get("best_match")
            results = search_resp.get("results") or []
            score = float(search_resp.get("confidence", 0.0) or 0.0)

        c_label = label_confidence(score)
        r_type = response_type_for_score(score)

        if best and isinstance(best, dict):
            best = {**best, "match_type": "semantic"}

        if results and isinstance(results, list):
            results = [{**r, "match_type": "semantic"} for r in results if isinstance(r, dict)]

        speech = build_speech(best, results, c_label, r_type)

        return VoiceWebhookResponse(
            speech=speech,
            type=r_type if best else "none",
            confidence=round(score, 3),
            confidence_label=c_label,
            best_match=best,
            results=results,
        )

    except Exception as exc:
        print("voice_webhook error:", repr(exc))
        return VoiceWebhookResponse(
            speech="Sorry, I’m having trouble reaching the S. A. P. assistant right now. Please try again.",
            type="error",
            confidence=0.0,
            confidence_label="low",
            best_match=None,
            results=[],
        )


@router.get("/voice-webhook/health")
def voice_webhook_health():
    _load_aliases_once()
    return {
        "data_dir": DATA_DIR,
        "aliases_loaded": bool(_alias_map),
        "alias_load_error": _alias_load_error,
        "search_ready": bool(search_is_ready()) if callable(search_is_ready) else False,
        "alias_count": len(_alias_map),
    }