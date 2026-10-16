"""
Pydantic Schemas for Full Career Transition Analysis Result.
"""

from typing import Optional
from pydantic import BaseModel, Field

from source.domain.entities.analysis import AnalysisResult


class AnalysisDetailResponse(BaseModel):
    """Response containing complete career advisory metrics and roadmap."""
    session_id: str = Field(..., description="Unique analysis workflow thread ID")
    status: str = Field("COMPLETED", description="Status of the analysis")
    analysis: Optional[AnalysisResult] = Field(None, description="Detailed domain analysis entity")
