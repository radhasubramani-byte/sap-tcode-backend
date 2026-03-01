# app/voice_webhook.py
import os
import logging
import glob
import csv
from typing import Optional, Dict, Any, List, Tuple
from fastapi import APIRouter, Request, Header, HTTPException
from pydantic import BaseModel
import requests

logger = logging.getLogger("voice_webhook")
router = APIRouter()

# Config from environment
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
SEARCH_URL = os.environ.get("SEARCH_URL", "https://sap-tcode-backend.onrender.com/search-tcode")
# thresholds
HIGH_THRESHOLD = float(os.environ.get("VOICE_HIGH_THRESHOLD", 0.75))
MEDIUM_THRESHOLD = float(os.environ.get("VOICE_MEDIUM_THRESHOLD", 0.40))
GAP_REQUIRED = float(os.environ.get("VOICE_GAP_REQUIRED", 0.08))
# alias directory (where you uploaded the CSVs)
ALIASES_DIR = os.environ.get("ALIASES_DIR", "/app/data")

# runtime alias map (lazy loaded)
_alias_map: Dict[str, Dict[str, str]] = {}
_aliases_loaded = False

class VoiceHookRequest(BaseModel):
    session_id: str
    transcript: str


def _strip_bearer(header_value: Optional[str]) -> Optional[str]:
    if header_value is None:
        return None
    v = header_value.strip()
    if v.lower().startswith("bearer "):
        return v.split(" ", 1)[1].strip()
    return v


def normalize_transcript(text: str) -> str:
    if not text:
        return ""
    t = text.lower().strip()

    for prefix in (
        "how do i ",
        "how to ",
        "how do we ",
        "how can i ",
        "how can we ",
        "what is the ",
        "what's the ",
        "i want to ",
        "i need to ",
        "please ",
    ):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break

    for drop in (" in sap", " in the system", " please", " thanks", " thank you", " in sap system"):
        t = t.replace(drop, "")

    t = t.replace("purchase requisitions", "create purchase requisition")
    t = t.replace("purchase requisition", "create purchase requisition")
    t = t.replace("post vendor invoice", "post vendor invoice")
    t = t.replace("vendor invoice", "vendor invoice")

    t = " ".join(t.split())
    return t


def load_aliases_from_dir(dirpath: str = ALIASES_DIR) -> None:
    """
    Load alias CSV files from the data dir into _alias_map.
    CSV format expected: alias,tcode,canonical_desc
    """
    global _alias_map, _aliases_loaded
    if _aliases_loaded:
        return

    alias_files = []
    # accept various naming patterns
    patterns = ["*_aliases*.csv", "*-aliases*.csv", "*aliases*.csv"]
    for p in patterns:
        alias_files.extend(glob.glob(os.path.join(dirpath, p)))

    loaded = 0
    _alias_map = {}

    for path in alias_files:
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    # tolerate rows with extra spaces
                    # expect alias, tcode, canonical_desc
                    alias = row[0].strip().lower() if len(row) >= 1 else ""
                    tcode = row[1].strip() if len(row) >= 2 else ""
                    desc = row[2].strip() if len(row) >= 3 else ""
                    if alias:
                        _alias_map[alias] = {"tcode": tcode, "description": desc}
                        loaded += 1
        except FileNotFoundError:
            continue
        except Exception as exc:  # be defensive
            logger.exception("Failed to load alias file %s: %s", path, exc)

    _aliases_loaded = True
    logger.info("Loaded %d aliases from %d files (dir=%s)", len(_alias_map), len(alias_files), dirpath)


def find_best_alias_match(transcript: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Find the longest alias (by length) that appears as a substring in transcript.
    Returns (alias, metadata) or None.
    """
    if not transcript:
        return None
    if not _aliases_loaded:
        try:
            load_aliases_from_dir()
        except Exception:
            # log but continue
            logger.exception("Error loading alias files")

    if not _alias_map:
        return None

    txt = transcript.lower()
    # find all aliases that are contained in the transcript
    matches = []
    for alias in _alias_map.keys():
        if alias and alias in txt:
            matches.append(alias)

    if not matches:
        return None

    # choose the longest alias (prefer more specific)
    best_alias = max(matches, key=len)
    return best_alias, _alias_map[best_alias]


def call_search_service_http(query: str, top_k: int = 5) -> Dict[str, Any]:
    try:
        params = {"q": query, "top_k": top_k}
        resp = requests.get(SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        logger.exception("search service HTTP request failed for query=%s", query)
        raise


def choose_best_from_search(resp: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "best_match": None,
        "results": [],
        "confidence": 0.0,
        "confidence_label": "low",
        "scores": [],
    }
    if not isinstance(resp, dict):
        return out

    if "best_match" in resp and resp.get("results") is not None:
        best = resp.get("best_match")
        results = resp.get("results", [])
        confidence = float(resp.get("confidence", 0.0)) if resp.get("confidence") is not None else 0.0
        out.update({"best_match": best, "results": results, "confidence": confidence})
        if resp.get("confidence_label"):
            out["confidence_label"] = resp["confidence_label"]
            return out

    results = resp.get("results") or resp.get("results_list") or []
    parsed_results: List[Dict[str, Any]] = []
    scores: List[float] = []
    for item in results:
        if isinstance(item, dict):
            parsed_results.append(item)
            s = item.get("score")
            try:
                scores.append(float(s) if s is not None else 0.0)
            except Exception:
                scores.append(0.0)

    out["results"] = parsed_results
    out["scores"] = scores

    if scores:
        best_score = scores[0]
        second = scores[1] if len(scores) > 1 else 0.0
        out["confidence"] = float(best_score)
        if best_score >= HIGH_THRESHOLD and (best_score - second) >= GAP_REQUIRED:
            out["confidence_label"] = "high"
        elif best_score >= MEDIUM_THRESHOLD:
            out["confidence_label"] = "medium"
        else:
            out["confidence_label"] = "low"

        out["best_match"] = parsed_results[0] if parsed_results else None
    else:
        out["confidence"] = 0.0
        out["confidence_label"] = "low"
        out["best_match"] = None

    return out


def build_speech_for_result(best: Optional[Dict[str, Any]], label: str, confidence: float) -> str:
    if label == "high":
        if best:
            tcode = best.get("tcode") or best.get("code") or best.get("id") or ""
            desc = best.get("description") or best.get("title") or ""
            if tcode:
                return f"I found {desc} — transaction {tcode}. Would you like me to open it for you?"
            else:
                return f"I found {desc}. Would you like me to open it for you?"
        return "I found a good match. Would you like me to open it for you?"
    elif label == "medium":
        if best:
            desc = best.get("description") or best.get("title") or ""
            tcode = best.get("tcode") or best.get("code") or ""
            if tcode:
                return f"It looks like you might mean {desc} (transaction {tcode}). Shall I proceed with that?"
            else:
                return f"It seems you might mean {desc}. Is that what you meant?"
        return "I found a few possible matches — can you tell me which one you mean?"
    else:
        return (
            "I need a little more detail to be sure which transaction you mean "
            "(for example, 'create purchase requisition' or 'post vendor invoice'). "
            "Can you rephrase or give an example?"
        )


@router.post("/voice-webhook")
async def voice_webhook(
    req: VoiceHookRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    provided = _strip_bearer(authorization)
    if not WEBHOOK_SECRET:
        logger.warning("WEBHOOK_SECRET not configured in environment")
    if WEBHOOK_SECRET and provided != WEBHOOK_SECRET:
        logger.warning("Invalid webhook secret provided=%s configured_exists=%s", provided, bool(WEBHOOK_SECRET))
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    raw = req.transcript or ""
    norm = normalize_transcript(raw)
    logger.info("voice_webhook: session=%s raw=%r norm=%r", req.session_id, raw, norm)

    if not norm:
        speech = "I didn't catch that — could you say it again?"
        return {
            "speech": speech,
            "type": "clarification",
            "confidence": 0.0,
            "confidence_label": "low",
            "best_match": None,
            "results": [],
        }

    # 1) Alias shortcut (highest priority)
    try:
        alias_match = find_best_alias_match(norm)
    except Exception:
        alias_match = None
        logger.exception("Alias matching failed for transcript=%s", norm)

    if alias_match:
        alias_text, meta = alias_match
        tcode = meta.get("tcode") or ""
        desc = meta.get("description") or ""
        best_match = {"tcode": tcode, "description": desc, "alias": alias_text}
        speech = f"I think you mean {desc} — transaction {tcode}. Should I open it for you?"
        out = {
            "speech": speech,
            "type": "confident",
            "confidence": 0.95,
            "confidence_label": "high",
            "best_match": best_match,
            "results": [best_match],
            "alias_matched": alias_text,
        }
        logger.info("Alias matched: %s -> %s", alias_text, best_match)
        return out

    # 2) No alias -> call search service
    try:
        search_resp = call_search_service_http(norm, top_k=5)
    except Exception:
        logger.exception("search service failure for query=%s", norm)
        speech = "Sorry, I'm having trouble reaching the SAP knowledge system right now."
        return {
            "speech": speech,
            "type": "error",
            "confidence": 0.0,
            "confidence_label": "low",
            "best_match": None,
            "results": [],
        }

    chosen = choose_best_from_search(search_resp)
    label = chosen.get("confidence_label", "low")
    confidence = float(chosen.get("confidence") or 0.0)
    speech = build_speech_for_result(chosen.get("best_match"), label, confidence)

    out = {
        "speech": speech,
        "type": "confident" if label in ("high", "medium") else "clarification",
        "confidence": round(confidence, 4),
        "confidence_label": label,
        "best_match": chosen.get("best_match"),
        "results": chosen.get("results"),
    }
    logger.debug("voice_webhook response: %s", out)
    return out