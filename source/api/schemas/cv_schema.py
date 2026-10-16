"""
Pydantic Schemas for CV Ingestion and Human-in-the-Loop Confirmation.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from source.domain.entities.candidate import CandidateProfile


class AnalyzeCVResponse(BaseModel):
    """Response returned after Phase 1 CV Ingestion, paused at HITL checkpoint."""
    session_id: str = Field(..., description="Unique analysis workflow thread ID")
    status: str = Field("AWAITING_HUMAN_REVIEW", description="Workflow state (e.g. AWAITING_HUMAN_REVIEW)")
    candidate_profile: Optional[CandidateProfile] = Field(None, description="Extracted candidate structured resume data")
    suggested_soc_code: Optional[str] = Field(None, description="Matched O*NET SOC code suggestion")
    suggested_title: Optional[str] = Field(None, description="Official O*NET occupation title")
    match_confidence: float = Field(0.0, description="Confidence score [0.0, 1.0] of title match")
    latency_ms: float = Field(0.0, description="Ingestion & parsing execution time in milliseconds")


class ConfirmAnalysisRequest(BaseModel):
    """Payload to confirm or override candidate title/SOC before deep ML analysis."""
    session_id: str = Field(..., description="Session thread ID from Phase 1")
    user_confirmed: bool = Field(True, description="True if user accepts suggested SOC code or provides valid override")
    overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional overrides e.g. {'matched_soc_code': '15-1252.00', 'matched_title': 'Cloud Architect'}",
    )
