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

Author: OpenAI
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Search service resolver
# -----------------------------------------------------------------------------
def _resolve_search_function() -> Callable[[str], Dict[str, Any]]:
    """
    Tries to find a usable search function inside the existing search_service.py.

    Supported function names:
    - search_tcodes
    - search_tcode
    - search
    - find_tcode
    - find_tcodes

    Raises:
        ImportError: If search_service.py or a usable function is not found.
    """
    try:
        import search_service  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Could not import search_service.py. Make sure it exists in the same project."
        ) from exc

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
        "search_service.py was found, but no supported search function was detected. "
        "Expected one of: search_tcodes, search_tcode, search, find_tcode, find_tcodes"
    )


# Resolve once at import time.
SEARCH_FUNCTION = _resolve_search_function()


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _safe_str(value: Any) -> str:
    """Convert a value to a safe stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def _normalize_result(raw_result: Any) -> Dict[str, Any]:
    """
    Normalize whatever the underlying search service returns into a consistent dict.

    Expected normalized output:
    {
        "type": "...",
        "confidence": 0.0,
        "confidence_label": "low|medium|high",
        "best_match": {...} | None,
        "results": [...]
    }
    """
    if raw_result is None:
        return {
            "type": "no_match",
            "confidence": 0.0,
            "confidence_label": "low",
            "best_match": None,
            "results": [],
        }

    if isinstance(raw_result, dict):
        result = dict(raw_result)
    else:
        # If some custom object is returned, try best-effort extraction.
        result = {
            "type": getattr(raw_result, "type", "no_match"),
            "confidence": getattr(raw_result, "confidence", 0.0),
            "confidence_label": getattr(raw_result, "confidence_label", "low"),
            "best_match": getattr(raw_result, "best_match", None),
            "results": getattr(raw_result, "results", []),
        }

    result.setdefault("type", "no_match")
    result.setdefault("confidence", 0.0)
    result.setdefault("confidence_label", "low")
    result.setdefault("best_match", None)
    result.setdefault("results", [])

    # Normalize confidence
    try:
        result["confidence"] = float(result.get("confidence", 0.0) or 0.0)
    except Exception:
        result["confidence"] = 0.0

    # Normalize confidence label
    confidence_label = _safe_str(result.get("confidence_label", "low")).lower()
    if confidence_label not in {"low", "medium", "high"}:
        confidence_label = "low"
    result["confidence_label"] = confidence_label

    # Normalize best_match
    best_match = result.get("best_match")
    if best_match is not None and not isinstance(best_match, dict):
        try:
            best_match = dict(best_match)
        except Exception:
            best_match = {
                "tcode": getattr(best_match, "tcode", ""),
                "description": getattr(best_match, "description", ""),
                "module": getattr(best_match, "module", ""),
            }
    result["best_match"] = best_match

    # Normalize results
    raw_results = result.get("results", [])
    normalized_results: List[Dict[str, Any]] = []
    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict):
                normalized_results.append(item)
            else:
                normalized_results.append(
                    {
                        "tcode": getattr(item, "tcode", ""),
                        "description": getattr(item, "description", ""),
                        "module": getattr(item, "module", ""),
                        "score": getattr(item, "score", None),
                    }
                )
    result["results"] = normalized_results

    return result


def _normalize_match_item(item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize a single match item to a consistent structure.
    Handles common field name variations.
    """
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

    return {
        "tcode": _safe_str(tcode),
        "description": _safe_str(description),
        "module": _safe_str(module),
        "score": score,
    }


def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates by T-code, keeping first occurrence."""
    seen = set()
    deduped: List[Dict[str, Any]] = []

    for item in results:
        normalized = _normalize_match_item(item)
        key = normalized["tcode"].upper()

        if not key:
            # Keep items without T-code only if needed
            key = f"__NO_TCODE__::{normalized['description']}::{normalized['module']}"

        if key in seen:
            continue

        seen.add(key)
        deduped.append(normalized)

    return deduped


def _looks_irrelevant_query(user_message: str) -> bool:
    """
    Lightweight guard for obviously irrelevant/non-SAP queries.
    This is intentionally conservative so it does NOT block valid SAP requests.
    """
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


# -----------------------------------------------------------------------------
# Chat response builders
# -----------------------------------------------------------------------------
def _build_high_confidence_reply(best_match: Dict[str, Any]) -> str:
    tcode = best_match["tcode"]
    description = best_match["description"]
    module = best_match["module"]

    lines = [
        f"SAP T-code: {tcode}",
        f"Description: {description or 'N/A'}",
    ]

    if module:
        lines.append(f"Module: {module}")

    return "\n".join(lines)


def _build_medium_confidence_reply(
    best_match: Dict[str, Any],
    alternatives: List[Dict[str, Any]],
    confidence_label: str,
) -> str:
    tcode = best_match["tcode"]
    description = best_match["description"]
    module = best_match["module"]

    lines = [
        f"Best match SAP T-code: {tcode}",
        f"Description: {description or 'N/A'}",
    ]

    if module:
        lines.append(f"Module: {module}")

    lines.append(f"Confidence: {confidence_label.title()}")

    if alternatives:
        lines.append("")
        lines.append("Other possible matches:")
        for item in alternatives[:3]:
            alt_tcode = item["tcode"]
            alt_desc = item["description"] or "No description"
            alt_module = item["module"]
            suffix = f" ({alt_module})" if alt_module else ""
            lines.append(f"- {alt_tcode} — {alt_desc}{suffix}")

    return "\n".join(lines)


def _build_multiple_matches_reply(results: List[Dict[str, Any]]) -> str:
    lines = ["I found multiple possible SAP T-codes:"]

    for item in results[:5]:
        tcode = item["tcode"]
        description = item["description"] or "No description"
        module = item["module"]
        suffix = f" ({module})" if module else ""
        lines.append(f"- {tcode} — {description}{suffix}")

    lines.append("")
    lines.append("Please rephrase the business task more specifically.")

    return "\n".join(lines)


def _build_no_match_reply(results: List[Dict[str, Any]]) -> str:
    if results:
        lines = [
            "I could not find a strong SAP T-code match for that request.",
            "",
            "Closest matches:",
        ]
        for item in results[:5]:
            tcode = item["tcode"]
            description = item["description"] or "No description"
            module = item["module"]
            suffix = f" ({module})" if module else ""
            lines.append(f"- {tcode} — {description}{suffix}")

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


def build_chat_reply(result: Dict[str, Any], user_message: str) -> str:
    """
    Create the chat-friendly reply text from normalized search result.
    """
    if _looks_irrelevant_query(user_message):
        return _build_irrelevant_reply()

    result_type = _safe_str(result.get("type", "no_match")).lower()
    confidence_label = _safe_str(result.get("confidence_label", "low")).lower()

    best_match = _normalize_match_item(result.get("best_match"))
    results = _dedupe_results(result.get("results", []))

    has_best_match = bool(best_match.get("tcode"))

    # High-confidence direct answer
    if result_type == "match" and has_best_match and confidence_label == "high":
        return _build_high_confidence_reply(best_match)

    # Match but medium/low confidence
    if result_type == "match" and has_best_match:
        alternatives = [r for r in results if r.get("tcode") != best_match.get("tcode")]
        return _build_medium_confidence_reply(best_match, alternatives, confidence_label)

    # Underlying search engine may return a best_match even when type differs
    if has_best_match and confidence_label == "high":
        return _build_high_confidence_reply(best_match)

    if result_type in {"multiple_matches", "multiple", "ambiguous"} and results:
        return _build_multiple_matches_reply(results)

    if has_best_match:
        alternatives = [r for r in results if r.get("tcode") != best_match.get("tcode")]
        return _build_medium_confidence_reply(best_match, alternatives, confidence_label)

    return _build_no_match_reply(results)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
def search_tcodes_for_chat(user_message: str) -> Dict[str, Any]:
    """
    Main entry point for the chat webhook.

    Returns:
        {
            "reply": "...",
            "type": "...",
            "confidence": 0.0,
            "confidence_label": "...",
            "best_match": {...} | None,
            "results": [...]
        }
    """
    query = _safe_str(user_message)

    if not query:
        return {
            "reply": "Please enter an SAP business task or an SAP T-code-related question.",
            "type": "no_match",
            "confidence": 0.0,
            "confidence_label": "low",
            "best_match": None,
            "results": [],
        }

    try:
        raw_result = SEARCH_FUNCTION(query)
        normalized = _normalize_result(raw_result)

        # Normalize best_match and results for output consistency
        best_match = _normalize_match_item(normalized.get("best_match"))
        best_match_output: Optional[Dict[str, Any]] = best_match if best_match.get("tcode") else None

        results = _dedupe_results(normalized.get("results", []))

        reply = build_chat_reply(
            {
                "type": normalized.get("type", "no_match"),
                "confidence": normalized.get("confidence", 0.0),
                "confidence_label": normalized.get("confidence_label", "low"),
                "best_match": best_match_output,
                "results": results,
            },
            query,
        )

        return {
            "reply": reply,
            "type": normalized.get("type", "no_match"),
            "confidence": float(normalized.get("confidence", 0.0) or 0.0),
            "confidence_label": normalized.get("confidence_label", "low"),
            "best_match": best_match_output,
            "results": results,
        }

    except Exception as exc:
        logger.exception("Chat search failed for query: %s", query)
        return {
            "reply": (
                "The SAP chat assistant encountered an internal error while searching.\n"
                "Please try again or contact the support team."
            ),
            "type": "error",
            "confidence": 0.0,
            "confidence_label": "low",
            "best_match": None,
            "results": [],
            "error": str(exc),
        }