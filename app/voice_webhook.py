# app/voice_webhook.py

"""
Voice webhook router for SAP T-code Assistant

Features
- deterministic alias matching
- semantic search fallback
- confidence handling
- robust DATA_DIR detection
- TTS optimized speech for demos
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

try:
    from app.services.search_service import search_tcode, is_ready as search_is_ready
except Exception:
    search_tcode = None
    search_is_ready = lambda: False


router = APIRouter()


# -----------------------------
# Request / Response models
# -----------------------------

class VoiceWebhookRequest(BaseModel):
    session_id: str = Field(default="unknown")
    transcript: str = Field(default="")


class VoiceWebhookResponse(BaseModel):
    speech: str
    type: str = "error"
    confidence: float = 0.0
    confidence_label: str = "low"
    best_match: Optional[Dict] = None
    results: List[Dict] = []


# -----------------------------
# Data directory detection
# -----------------------------

def detect_data_dir():

    candidates = [
        os.getenv("DATA_DIR"),
        "app/data",
        "/app/app/data",
        "/app/data"
    ]

    for p in candidates:
        if p and os.path.isdir(p):
            return p

    return None


DATA_DIR = detect_data_dir()


# -----------------------------
# Alias structure
# -----------------------------

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
_alias_loaded = False


# -----------------------------
# Text normalization
# -----------------------------

def _normalize_text(s: str) -> str:

    if not s:
        return ""

    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)

    return s


# -----------------------------
# Load aliases
# -----------------------------

def _load_aliases_once():

    global _alias_loaded, _alias_map

    if _alias_loaded:
        return

    _alias_loaded = True

    if not DATA_DIR:
        print("DATA_DIR not found")
        return

    for filename, module, priority in _ALIAS_FILES_PRIORITY:

        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            continue

        try:

            with open(path, "r", encoding="utf-8") as f:

                reader = csv.DictReader(f)

                for row in reader:

                    alias_raw = row.get("alias", "").strip()
                    tcode = row.get("tcode", "").strip()
                    desc = row.get("canonical_desc", "").strip()

                    if not alias_raw or not tcode:
                        continue

                    key = _normalize_text(alias_raw)

                    rec = AliasRecord(
                        alias=alias_raw,
                        tcode=tcode,
                        canonical_desc=desc,
                        module=module,
                        priority=priority
                    )

                    prev = _alias_map.get(key)

                    if prev is None or rec.priority < prev.priority:
                        _alias_map[key] = rec

            print(f"Loaded alias file {filename}")

        except Exception as exc:
            print("Alias load error:", exc)


# -----------------------------
# Alias matching
# -----------------------------

def match_alias(transcript: str):

    _load_aliases_once()

    key = _normalize_text(transcript)

    rec = _alias_map.get(key)

    if not rec:
        return None

    return {
        "tcode": rec.tcode,
        "description": rec.canonical_desc,
        "module": rec.module,
        "score": 1.0
    }


# -----------------------------
# Confidence thresholds
# -----------------------------

CONFIDENT_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.50


def label_confidence(score):

    if score >= CONFIDENT_THRESHOLD:
        return "high"

    if score >= MEDIUM_THRESHOLD:
        return "medium"

    return "low"


def response_type(score):

    return "confident" if score >= MEDIUM_THRESHOLD else "uncertain"


# -----------------------------
# Build speech (TTS optimized)
# -----------------------------

def build_speech(best_match, results, confidence_label, r_type):

    if not best_match:

        return (
            "I couldn’t find a matching S-A-P transaction code. "
            "Try saying something like create purchase order or change material."
        )

    tcode = best_match.get("tcode")
    desc = best_match.get("description", "")
    module = best_match.get("module", "")

    # ---------------------------------------------------------
    # TINY 2-LINE DEMO IMPROVEMENT (forces clear TTS pause)
    # ---------------------------------------------------------

    if module:

        return f"The S-A-P T-code is… {tcode}. {desc}. Module {module}."

    return f"The S-A-P T-code is… {tcode}. {desc}."


# -----------------------------
# Authentication
# -----------------------------

def check_token(auth_header):

    expected = (
        os.getenv("VOICE_WEBHOOK_TOKEN")
        or os.getenv("SAP_VOICE_WEBHOOK_TOKEN")
        or os.getenv("WEBHOOK_SECRET")
    )

    if not expected:
        return

    if not auth_header:
        raise HTTPException(status_code=401)

    if not auth_header.lower().startswith("bearer"):
        raise HTTPException(status_code=401)

    token = auth_header.split(" ")[1]

    if token != expected:
        raise HTTPException(status_code=403)


# -----------------------------
# Main webhook
# -----------------------------

@router.post("/voice-webhook", response_model=VoiceWebhookResponse)
def voice_webhook(payload: VoiceWebhookRequest, authorization: Optional[str] = Header(None)):

    check_token(authorization)

    transcript = payload.transcript.strip()

    if not transcript:

        return VoiceWebhookResponse(
            speech="Please tell me what you want to do in S-A-P.",
            type="none"
        )

    # ------------------------
    # Alias match first
    # ------------------------

    alias_hit = match_alias(transcript)

    if alias_hit:

        speech = build_speech(alias_hit, [alias_hit], "high", "confident")

        return VoiceWebhookResponse(
            speech=speech,
            type="confident",
            confidence=1.0,
            confidence_label="high",
            best_match=alias_hit,
            results=[alias_hit]
        )

    # ------------------------
    # Semantic search
    # ------------------------

    if not callable(search_tcode):

        return VoiceWebhookResponse(
            speech="Search service unavailable",
            type="error"
        )

    if not search_is_ready():

        return VoiceWebhookResponse(
            speech="One moment. Loading the S-A-P knowledge base.",
            type="warming_up"
        )

    resp = search_tcode(transcript, top_k=5)

    best = resp.get("best_match")
    results = resp.get("results", [])
    score = float(resp.get("confidence", 0))

    label = label_confidence(score)
    r_type = response_type(score)

    speech = build_speech(best, results, label, r_type)

    return VoiceWebhookResponse(
        speech=speech,
        type=r_type,
        confidence=round(score, 3),
        confidence_label=label,
        best_match=best,
        results=results
    )