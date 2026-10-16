"""
Pydantic Schemas for Interactive Copilot Chat.
"""

from typing import Optional
import time
from pydantic import BaseModel, Field


class CopilotChatRequest(BaseModel):
    """Payload for conversational question to AI Career Copilot."""
    session_id: str = Field(..., description="Active session ID with completed analysis")
    message: str = Field(..., min_length=1, description="User follow-up inquiry")


class CopilotChatResponse(BaseModel):
    """Standard JSON response for Copilot chat (non-streaming variant)."""
    session_id: str = Field(..., description="Session thread ID")
    response: str = Field(..., description="Copilot answer grounded in O*NET evidence")
    flow_type: Optional[str] = Field(default=None, description="Dispatched interaction flow: DIRECT_CHAT | KNOWLEDGE_RAG | WORKFLOW")
    latency_ms: Optional[float] = Field(default=None, description="Response latency in milliseconds")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of message generation")
