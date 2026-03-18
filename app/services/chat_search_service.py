"""
chat_search_service.py

Separate chat-only search/formatting layer for the SAP T-Code Assistant.
This file does NOT modify the existing voice flow.

Purpose:
- Reuse the same backend search logic
- Build chat-friendly text responses
- Keep chat isolated from the voice agent

Expected behavior:
Input:  "How do I create a purchase order?"
Output: "SAP T-code: ME21N\nDescription: Create Purchase Order\nModule: MM"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Callable

from app.services import search_service

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Search service resolver
# -----------------------------------------------------------------------------
def _resolve_search_function() -> Callable[..., Dict[str, Any]]:
    """
    Tries to find a usable search function inside the existing search_service.py.

    Supported function names:
    - search_tcodes
    - search_tcode
    - search
    - find_tcode
    - find_tcodes
    """
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
            logger.info("Using search function from search_service: %s", name)
            return fn

    raise ImportError(
        "app.services.search_service was found, but no supported search function was detected. "
        "Expected one of: search_tcodes, search_tcode, search, find_tcode, find_tcodes"
    )


SEARCH_FUNCTION = _resolve_search_function()


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_match_item(item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not item:
        return {"tcode": "", "description": "", "module": "", "score": None}

    tcode = (
        item.get("tcode")
        or item.get("code")
        or item.get("transaction_code")
        or item.get("sap_tcode")
        or ""
    )
    description = (
        item.get("description")
        or item.get("desc")
        or item.get("text")
        or item.get("title")
        or ""
    )
    module = item.get("module") or item.get("component") or item.get("area") or ""
    score = item.get("score", item.get("similarity"))

    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None

    return {
        "tcode": _safe_str(tcode),
        "description": _safe_str(description),
        "module": _safe_str(module),
        "score": score,
    }


def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []

    for item in results:
        normalized = _normalize_match_item(item)
        key = normalized["tcode"].upper()

        if not key:
            key = f"__NO_TCODE__::{normalized['description']}::{normalized['module']}"

        if key in seen:
            continue

        seen.add(key)
        deduped.append(normalized)

    return deduped


def _looks_irrelevant_query(user_message: str) -> bool:
    text = _safe_str(user_message).lower()
    if not text:
        return True

    irrelevant_signals = [
        "weather",
        "temperature",
        "tell me a joke",
        "joke",
        "news",
        "who is the president",
        "stock price",
        "movie showtimes",
        "sports score",
    ]

    sap_signals = [
        "sap",
        "tcode",
        "t-code",
        "purchase order",
        "sales order",
        "goods receipt",
        "material",
        "invoice",
        "delivery",
        "sto",
        "vendor",
        "customer",
        "production order",
        "billing",
        "gr",
        "po",
        "pr",
    ]

    has_irrelevant = any(signal in text for signal in irrelevant_signals)
    has_sap_context = any(signal in text for signal in sap_signals)

    return has_irrelevant and not has_sap_context


def _score_to_confidence_label(score: Optional[float]) -> str:
    if score is None:
        return "low"
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def _normalize_result(raw_result: Any) -> Dict[str, Any]:
    """
    Normalizes the actual response format coming from app.services.search_service.

    Expected real shapes from your backend:
    - {"type": "confident"|"uncertain"|"none", ...}
    - {"status": "warming_up"|"error"|"invalid_query", ...}
    """
    if raw_result is None:
        return {
            "type": "none",
            "status": None,
            "message": "",
            "confidence": 0.0,
            "confidence_label": "low",
            "best_match": None,
            "alternatives": [],
            "results": [],
        }

    if isinstance(raw_result, dict):
        result = dict(raw_result)
    else:
        result = {
            "type": getattr(raw_result, "type", "none"),
            "status": getattr(raw_result, "status", None),
            "message": getattr(raw_result, "message", ""),
            "confidence": getattr(raw_result, "confidence", 0.0),
            "confidence_label": getattr(raw_result, "confidence_label", "low"),
            "best_match": getattr(raw_result, "best_match", None),
            "alternatives": getattr(raw_result, "alternatives", []),
            "results": getattr(raw_result, "results", []),
        }

    result.setdefault("type", "none")
    result.setdefault("status", None)
    result.setdefault("message", "")
    result.setdefault("confidence", 0.0)
    result.setdefault("confidence_label", "low")
    result.setdefault("best_match", None)
    result.setdefault("alternatives", [])
    result.setdefault("results", [])

    try:
        result["confidence"] = float(result.get("confidence", 0.0) or 0.0)
    except Exception:
        result["confidence"] = 0.0

    confidence_label = _safe_str(result.get("confidence_label", "")).lower()
    if confidence_label not in {"low", "medium", "high"}:
        confidence_label = _score_to_confidence_label(result.get("confidence"))
    result["confidence_label"] = confidence_label

    result["best_match"] = _normalize_match_item(result.get("best_match"))

    normalized_results: List[Dict[str, Any]] = []
    for item in result.get("results", []) or []:
        normalized_results.append(_normalize_match_item(item))
    result["results"] = _dedupe_results(normalized_results)

    normalized_alternatives: List[Dict[str, Any]] = []
    for item in result.get("alternatives", []) or []:
        normalized_alternatives.append(_normalize_match_item(item))
    result["alternatives"] = _dedupe_results(normalized_alternatives)

    return result


# -----------------------------------------------------------------------------
# Chat response builders
# -----------------------------------------------------------------------------
def _build_high_confidence_reply(best_match: Dict[str, Any]) -> str:
    lines = [
        f"SAP T-code: {best_match['tcode']}",
        f"Description: {best_match['description'] or 'N/A'}",
    ]

    if best_match["module"]:
        lines.append(f"Module: {best_match['module']}")

    return "\n".join(lines)


def _build_uncertain_reply(
    best_match: Dict[str, Any],
    alternatives: List[Dict[str, Any]],
    confidence_label: str,
) -> str:
    lines = [
        f"Best match SAP T-code: {best_match['tcode']}",
        f"Description: {best_match['description'] or 'N/A'}",
    ]

    if best_match["module"]:
        lines.append(f"Module: {best_match['module']}")

    lines.append(f"Confidence: {confidence_label.title()}")

    if alternatives:
        lines.append("")
        lines.append("Other possible matches:")
        for item in alternatives[:3]:
            suffix = f" ({item['module']})" if item["module"] else ""
            lines.append(f"- {item['tcode']} — {item['description'] or 'No description'}{suffix}")

    return "\n".join(lines)


def _build_no_match_reply(results: List[Dict[str, Any]]) -> str:
    if results:
        lines = [
            "I could not find a strong SAP T-code match for that request.",
            "",
            "Closest matches:",
        ]
        for item in results[:5]:
            suffix = f" ({item['module']})" if item["module"] else ""
            lines.append(f"- {item['tcode']} — {item['description'] or 'No description'}{suffix}")

        lines.append("")
        lines.append("Try rephrasing the SAP business action.")
        return "\n".join(lines)

    return (
        "I could not find a matching SAP T-code.\n\n"
        "Try questions like:\n"
        "- create purchase order\n"
        "- post goods receipt\n"
        "- change material\n"
        "- create sales order"
    )


def _build_irrelevant_reply() -> str:
    return (
        "I can help only with SAP business tasks and SAP T-codes.\n\n"
        "Try questions like:\n"
        "- create purchase order\n"
        "- post goods receipt\n"
        "- change material\n"
        "- create sales order"
    )


def _build_status_reply(status: str, message: str) -> str:
    status = _safe_str(status).lower()

    if status == "warming_up":
        return "The SAP knowledge base is still loading. Please try again in a moment."

    if status == "invalid_query":
        return "Please enter an SAP business task or an SAP T-code-related question."

    if status == "error":
        return message or "The SAP chat assistant encountered an internal error while searching."

    return message or "The SAP chat assistant could not process the request."


def build_chat_reply(result: Dict[str, Any], user_message: str) -> str:
    if _looks_irrelevant_query(user_message):
        return _build_irrelevant_reply()

    status = _safe_str(result.get("status"))
    if status:
        return _build_status_reply(status, _safe_str(result.get("message")))

    result_type = _safe_str(result.get("type", "none")).lower()
    confidence_label = _safe_str(result.get("confidence_label", "low")).lower()

    best_match = _normalize_match_item(result.get("best_match"))
    results = _dedupe_results(result.get("results", []))
    alternatives = _dedupe_results(result.get("alternatives", []))

    has_best_match = bool(best_match.get("tcode"))

    if result_type == "confident" and has_best_match:
        return _build_high_confidence_reply(best_match)

    if result_type == "uncertain" and has_best_match:
        if not alternatives:
            alternatives = [r for r in results if r.get("tcode") != best_match.get("tcode")]
        return _build_uncertain_reply(best_match, alternatives, confidence_label)

    if result_type == "none":
        return _build_no_match_reply(results)

    if has_best_match:
        if not alternatives:
            alternatives = [r for r in results if r.get("tcode") != best_match.get("tcode")]
        return _build_uncertain_reply(best_match, alternatives, confidence_label)

    return _build_no_match_reply(results)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
def search_tcodes_for_chat(user_message: str) -> Dict[str, Any]:
    query = _safe_str(user_message)

    if not query:
        return {
            "reply": "Please enter an SAP business task or an SAP T-code-related question.",
            "type": "none",
            "status": "invalid_query",
            "confidence": 0.0,
            "confidence_label": "low",
            "best_match": None,
            "results": [],
            "alternatives": [],
        }

    try:
        try:
            raw_result = SEARCH_FUNCTION(query, top_k=5)
        except TypeError:
            raw_result = SEARCH_FUNCTION(query)

        normalized = _normalize_result(raw_result)

        best_match_output: Optional[Dict[str, Any]] = (
            normalized["best_match"] if normalized["best_match"].get("tcode") else None
        )

        results = normalized.get("results", [])
        alternatives = normalized.get("alternatives", [])

        if best_match_output and not alternatives:
            alternatives = [r for r in results if r.get("tcode") != best_match_output.get("tcode")]

        reply = build_chat_reply(
            {
                "type": normalized.get("type", "none"),
                "status": normalized.get("status"),
                "message": normalized.get("message", ""),
                "confidence": normalized.get("confidence", 0.0),
                "confidence_label": normalized.get("confidence_label", "low"),
                "best_match": best_match_output,
                "results": results,
                "alternatives": alternatives,
            },
            query,
        )

        return {
            "reply": reply,
            "type": normalized.get("type", "none"),
            "status": normalized.get("status"),
            "confidence": float(normalized.get("confidence", 0.0) or 0.0),
            "confidence_label": normalized.get("confidence_label", "low"),
            "best_match": best_match_output,
            "results": results,
            "alternatives": alternatives,
        }

    except Exception as exc:
        logger.exception("Chat search failed for query: %s", query)
        return {
            "reply": (
                "The SAP chat assistant encountered an internal error while searching.\n"
                "Please try again or contact the support team."
            ),
            "type": "none",
            "status": "error",
            "confidence": 0.0,
            "confidence_label": "low",
            "best_match": None,
            "results": [],
            "alternatives": [],
            "error": str(exc),
        }