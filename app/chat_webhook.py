"""
chat_webhook.py

Separate chat-only FastAPI router for the SAP T-Code Assistant.
This file does NOT change the existing voice agent.

Usage:
- Include this router in main.py
- POST to /chat/search-tcode

Example request:
{
  "session_id": "demo-123",
  "message": "How do I create a purchase order?"
}
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.chat_search_service import search_tcodes_for_chat

logger = logging.getLogger(__name__)

router = APIRouter(tags=["SAP Chat Agent"])


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Optional chat session ID")
    message: str = Field(..., description="User's SAP business task or question")


class TCodeResult(BaseModel):
    tcode: str = ""
    description: str = ""
    module: str = ""
    score: Optional[float] = None


class ChatResponse(BaseModel):
    reply: str
    type: str
    status: Optional[str] = None
    confidence: float
    confidence_label: str
    best_match: Optional[TCodeResult] = None
    results: List[TCodeResult] = []
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@router.get("/chat/health", response_model=HealthResponse)
def chat_health() -> HealthResponse:
    """
    Simple health endpoint for the chat agent.
    """
    return HealthResponse(
        status="ok",
        service="sap-tcode-chat-agent",
    )


@router.post("/chat/search-tcode", response_model=ChatResponse)
def chat_search_tcode(payload: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for SAP T-code lookup.

    Request:
    {
      "session_id": "abc123",
      "message": "How do I create a purchase order?"
    }

    Response:
    {
      "reply": "SAP T-code: ME21N\\nDescription: Create Purchase Order\\nModule: MM",
      "type": "confident",
      "status": null,
      "confidence": 0.97,
      "confidence_label": "high",
      "best_match": {...},
      "results": [...]
    }
    """
    try:
        user_message = (payload.message or "").strip()

        if not user_message:
            return ChatResponse(
                reply="Please enter an SAP business task or an SAP T-code-related question.",
                type="none",
                status="invalid_query",
                confidence=0.0,
                confidence_label="low",
                best_match=None,
                results=[],
                error=None,
            )

        result = search_tcodes_for_chat(user_message)

        best_match_obj = None
        if isinstance(result.get("best_match"), dict):
            best_match_obj = TCodeResult(**result["best_match"])

        result_items: List[TCodeResult] = []
        for item in result.get("results", []):
            if isinstance(item, dict):
                result_items.append(TCodeResult(**item))

        return ChatResponse(
            reply=str(result.get("reply", "")),
            type=str(result.get("type", "none")),
            status=result.get("status"),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            confidence_label=str(result.get("confidence_label", "low")),
            best_match=best_match_obj,
            results=result_items,
            error=result.get("error"),
        )

    except Exception as exc:
        logger.exception("Unhandled exception in /chat/search-tcode")
        raise HTTPException(
            status_code=500,
            detail=f"Chat search failed: {str(exc)}"
        ) from exc