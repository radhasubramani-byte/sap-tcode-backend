# app/voice_webhook.py
"""
Voice webhook router for SAP T-code Assistant.

Goals:
- Production-safe (no crashes at import time)
- Deterministic alias priority (exact match first, then semantic search)
- Clear confidence threshold logic
- Works on Render + local Windows (robust DATA_DIR detection)
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

# Prefer calling search_service directly (no HTTP loopback / timeouts)
try:
    from app.services.search_service import search_tcode, is_ready as search_is_ready
except Exception:
    search_tcode = None  # type: ignore
    search_is_ready = lambda: False  # type: ignore


router = APIRouter()

# -------------------------
# Models (request/response)
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
    """
    Detect where CSVs live for both local Windows + Render Docker.

    Typical candidates:
    - env DATA_DIR (preferred)
    - local repo: app/data
    - Render Docker after COPY . /app: /app/app/data
    - Some layouts: /app/data
    """
    candidates = [
        os.getenv("DATA_DIR"),
        "app/data",
        "/app/app/data",
        "/app/data",
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
    ("mm_aliases.csv", "MM", 1),  # highest priority for purchasing phrases like "create po"
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
    """
    Normalize user speech and alias text into a deterministic key.
    - lowercase
    - trim
    - collapse whitespace
    - remove most punctuation
    """
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[\u2018\u2019\u201C\u201D]", "'", s)  # smart quotes -> '
    s = re.sub(r"[^a-z0-9\s\-\+]", " ", s)            # keep alnum/space/-/+
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_aliases_once() -> None:
    """
    Load aliases from *_aliases.csv with deterministic file priority.
    On duplicates, keep the higher-priority record.
    """
    global _alias_loaded, _alias_load_error, _alias_map

    if _alias_loaded:
        return

    _alias_loaded = True
    _alias_map = {}
    _alias_load_error = None

    if not DATA_DIR:
        _alias_load_error = "DATA_DIR not found"
        print("No alias files found (no valid DATA_DIR detected)")
        return

    any_loaded = False

    for filename, module, priority in _ALIAS_FILES_PRIORITY:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                # Expect headers: alias,tcode,canonical_desc
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
                    if prev is None or rec.priority < prev.priority:
                        _alias_map[key] = rec

            any_loaded = True
            print(f"✅ Loaded aliases from {filename}: {path}")

        except Exception as exc:
            _alias_load_error = f"Failed reading {filename}: {exc}"
            print(f"❌ Failed to load {filename}: {exc}")

    if not any_loaded:
        print(f"No alias files found (no *_aliases.csv in data dir). DATA_DIR={DATA_DIR}")


def match_alias(transcript: str) -> Optional[Dict]:
    """
    Deterministic alias priority:
    - Exact normalized match only (no fuzzy)
    """
    _load_aliases_once()
    key = _normalize_text(transcript)
    rec = _alias_map.get(key)
    if not rec:
        return None

    return {
        "tcode": rec.tcode,
        "description": rec.canonical_desc or None,
        "module": rec.module,
        "score": 1.0,  # alias exact match = certainty
        "match_type": "alias_exact",
        "alias": rec.alias,
    }


# -------------------------
# Confidence threshold logic
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


def build_speech(best_match: Optional[Dict], results: List[Dict], c_label: str, r_type: str) -> str:
    if not best_match:
        return (
            "I couldn’t find a matching SAP transaction code for that. "
            "Try saying what you want to do, like 'create purchase order' or 'display sales order'."
        )

    tcode = best_match.get("tcode") or ""
    desc = best_match.get("description") or ""
    module = best_match.get("module") or ""

    if r_type == "confident" and c_label in ("high", "medium"):
        # short, voice-friendly
        if module:
            return f"The SAP T-code is {tcode} — {desc} ({module})."
        return f"The SAP T-code is {tcode} — {desc}."
    else:
        # Uncertain: give top options
        options = []
        for r in results[:3]:
            if r.get("tcode"):
                d = r.get("description") or ""
                options.append(f"{r['tcode']} ({d})")
        opts = "; ".join(options) if options else tcode
        return (
            "I’m not fully confident. Here are the closest matches: "
            f"{opts}. Which one did you mean?"
        )


# -------------------------
# Auth (optional)
# -------------------------

def _check_bearer_token(authorization: Optional[str]) -> None:
    """
    If VOICE_WEBHOOK_TOKEN is set, enforce Authorization: Bearer <token>.
    If not set, allow (useful for local dev).
    """
    expected = os.getenv("VOICE_WEBHOOK_TOKEN") or os.getenv("SAP_VOICE_WEBHOOK_TOKEN")
    if not expected:
        return

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    got = authorization.split(" ", 1)[1].strip()
    if got != expected:
        raise HTTPException(status_code=403, detail="Invalid token")


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
            speech="Please say what you want to do in SAP, for example: 'create purchase order'.",
            type="none",
            confidence=0.0,
            confidence_label="low",
            best_match=None,
            results=[],
        )

    # 1) Alias exact match (deterministic priority)
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

    # If embeddings still warming, return warming_up (voice-friendly)
    try:
        if not search_is_ready():
            return VoiceWebhookResponse(
                speech="One moment — I’m loading the SAP knowledge base.",
                type="warming_up",
                confidence=0.0,
                confidence_label="low",
                best_match=None,
                results=[],
            )
    except Exception:
        # If readiness check fails, still attempt search below
        pass

    try:
        search_resp = search_tcode(transcript, top_k=5)

        # If the service returns warming_up dict, pass through cleanly
        if isinstance(search_resp, dict) and search_resp.get("status") == "warming_up":
            return VoiceWebhookResponse(
                speech="One moment — I’m loading the SAP knowledge base.",
                type="warming_up",
                confidence=0.0,
                confidence_label="low",
                best_match=None,
                results=[],
            )

        # Your search_service returns: type/best_match/results/confidence/confidence_label
        best = None
        results: List[Dict] = []

        if isinstance(search_resp, dict):
            best = search_resp.get("best_match")
            results = search_resp.get("results") or search_resp.get("results", [])

            # Normalize confidence from search_service (0..1)
            score = float(search_resp.get("confidence", 0.0) or 0.0)
        else:
            score = 0.0

        c_label = label_confidence(score)
        r_type = response_type_for_score(score)

        # Always include match_type so downstream logic can be deterministic
        if best and isinstance(best, dict):
            best = {**best, "match_type": "semantic"}
        if results and isinstance(results, list):
            new_results = []
            for r in results:
                if isinstance(r, dict):
                    new_results.append({**r, "match_type": "semantic"})
            results = new_results

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
        # Hard fail-safe
        print("voice_webhook error:", repr(exc))
        return VoiceWebhookResponse(
            speech="Sorry, I’m having trouble reaching the SAP knowledge system right now.",
            type="error",
            confidence=0.0,
            confidence_label="low",
            best_match=None,
            results=[],
        )


# Optional tiny health endpoint (useful for debugging)
@router.get("/voice-webhook/health")
def voice_webhook_health():
    _load_aliases_once()
    return {
        "data_dir": DATA_DIR,
        "aliases_loaded": bool(_alias_map),
        "alias_load_error": _alias_load_error,
        "search_ready": bool(search_is_ready()) if callable(search_is_ready) else False,
    }