"""
chat_search_service.py

Chat-only formatting layer for SAP T-Code Assistant.
Does NOT modify voice agent.

Key features:
- Reuses existing search_service
- Clean enterprise UI formatting
- Strong T-code visual emphasis
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Callable

from app.services import search_service

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Resolve search function dynamically
# -----------------------------------------------------------------------------
def _resolve_search_function() -> Callable[..., Dict[str, Any]]:
    candidate_names = [
        "search_tcodes",
        "search_tcode",
        "search",
        "find_tcode",
        "find_tcodes",
    ]

    for name in candidate_names:
        fn = getattr(search_service, name, None)
        if callable(fn):
            logger.info("Using search function: %s", name)
            return fn

    raise ImportError("No valid search function found in search_service")


SEARCH_FUNCTION = _resolve_search_function()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_str(value: Any) -> str:
    return str(value).strip() if value else ""


def _normalize_match_item(item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not item:
        return {"tcode": "", "description": "", "module": "", "score": None}

    return {
        "tcode": _safe_str(
            item.get("tcode")
            or item.get("code")
            or item.get("transaction_code")
        ),
        "description": _safe_str(
            item.get("description")
            or item.get("desc")
            or item.get("text")
        ),
        "module": _safe_str(
            item.get("module")
            or item.get("component")
            or item.get("area")
        ),
        "score": item.get("score"),
    }


def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []

    for r in results:
        key = r.get("tcode", "").upper()
        if key and key not in seen:
            seen.add(key)
            out.append(r)

    return out


def _looks_irrelevant_query(q: str) -> bool:
    q = q.lower()

    irrelevant = ["weather", "joke", "news", "stock price"]
    sap = ["sap", "po", "pr", "invoice", "material", "delivery"]

    return any(x in q for x in irrelevant) and not any(x in q for x in sap)


# -----------------------------------------------------------------------------
# Response builders
# -----------------------------------------------------------------------------
def _build_high_confidence_reply(best_match: Dict[str, Any]) -> str:
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "SAP T-CODE",
        f"{best_match['tcode']}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"{best_match['description'] or 'N/A'}",
    ]

    if best_match["module"]:
        lines.append(f"Module: {best_match['module']}")

    lines.append("")
    lines.append("Confidence: High")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def _build_uncertain_reply(
    best_match: Dict[str, Any],
    alternatives: List[Dict[str, Any]],
    confidence_label: str,
) -> str:

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "SAP T-CODE",
        f"{best_match['tcode']}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"{best_match['description'] or 'N/A'}",
    ]

    if best_match["module"]:
        lines.append(f"Module: {best_match['module']}")

    lines.append("")
    lines.append(f"Confidence: {confidence_label.title()}")

    if alternatives:
        lines.append("")
        lines.append("Other matches:")
        for alt in alternatives[:3]:
            suffix = f" ({alt['module']})" if alt["module"] else ""
            lines.append(f"- {alt['tcode']} — {alt['description']}{suffix}")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def _build_no_match_reply(results: List[Dict[str, Any]]) -> str:
    if results:
        lines = [
            "No strong SAP match found.",
            "",
            "Closest matches:",
        ]

        for r in results[:5]:
            lines.append(f"- {r['tcode']} — {r['description']} ({r['module']})")

        return "\n".join(lines)

    return (
        "No SAP T-code found.\n\n"
        "Try:\n"
        "- create purchase order\n"
        "- post goods receipt\n"
        "- change material"
    )


def _build_irrelevant_reply() -> str:
    return (
        "I can help only with SAP tasks.\n\n"
        "Try:\n"
        "- create purchase order\n"
        "- post goods receipt\n"
        "- change material"
    )


# -----------------------------------------------------------------------------
# Core builder
# -----------------------------------------------------------------------------
def build_chat_reply(result: Dict[str, Any], user_message: str) -> str:

    if _looks_irrelevant_query(user_message):
        return _build_irrelevant_reply()

    result_type = result.get("type", "")
    confidence_label = result.get("confidence_label", "low")

    best_match = _normalize_match_item(result.get("best_match"))
    results = _dedupe_results(result.get("results", []))

    has_match = bool(best_match.get("tcode"))

    if result_type == "confident" and has_match:
        return _build_high_confidence_reply(best_match)

    if has_match:
        alternatives = [r for r in results if r["tcode"] != best_match["tcode"]]
        return _build_uncertain_reply(best_match, alternatives, confidence_label)

    return _build_no_match_reply(results)


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------
def search_tcodes_for_chat(user_message: str) -> Dict[str, Any]:

    query = _safe_str(user_message)

    if not query:
        return {
            "reply": "Please enter an SAP question.",
            "type": "none",
            "confidence": 0.0,
            "confidence_label": "low",
            "best_match": None,
            "results": [],
        }

    try:
        try:
            raw = SEARCH_FUNCTION(query, top_k=5)
        except TypeError:
            raw = SEARCH_FUNCTION(query)

        best_match = _normalize_match_item(raw.get("best_match"))
        results = [_normalize_match_item(r) for r in raw.get("results", [])]

        reply = build_chat_reply(
            {
                "type": raw.get("type"),
                "confidence_label": raw.get("confidence_label"),
                "best_match": best_match,
                "results": results,
            },
            query,
        )

        return {
            "reply": reply,
            "type": raw.get("type"),
            "confidence": raw.get("confidence", 0.0),
            "confidence_label": raw.get("confidence_label"),
            "best_match": best_match,
            "results": results,
        }

    except Exception as e:
        logger.exception("Chat error")
        return {
            "reply": "Internal error in SAP assistant.",
            "type": "error",
            "confidence": 0.0,
            "confidence_label": "low",
            "best_match": None,
            "results": [],
        }