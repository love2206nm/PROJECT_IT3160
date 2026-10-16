"""
Agent State Schema for LangGraph Multi-Agent Copilot State Machine.
Defines central TypedDict conforming strictly to Domain Entities:
- CandidateProfile
- RiskProfile
- SafeTransitionRole, SkillGap
- CopilotRoadmap
- AnalysisResult
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.risk import RiskProfile
from source.domain.entities.transition import SafeTransitionRole, SkillGap
from source.domain.entities.roadmap import CopilotRoadmap
from source.domain.entities.analysis import AnalysisResult


class AgentState(TypedDict, total=False):
    """
    Central state container for the LangGraph Copilot Multi-Agent Graph.
    Preserves type safety by housing validated Domain Entities throughout all nodes.
    """

    # Ingestion Layer
    session_id: Optional[str]
    raw_cv_text: str
    file_path: Optional[str]
    candidate_profile: Optional[CandidateProfile]
    user_confirmed: bool

    # Matching & Routing Layer
    matched_soc_code: Optional[str]
    matched_title: Optional[str]
    match_confidence: float
    is_fresher_mode: bool

    # ML & Clustering Layer
    risk_profile: Optional[RiskProfile]
    safe_pivot_roles: Optional[List[SafeTransitionRole]]
    skill_gap: Optional[SkillGap]
    skill_gaps: Optional[List[SkillGap]]

    # RAG Grounding Layer
    grounding_evidence: Optional[List[Dict[str, Any]]]

    # Copilot & Critic Reflection Layer
    copilot_roadmap: Optional[CopilotRoadmap]
    critic_review: Optional[Dict[str, Any]]
    critic_approved: bool
    critic_feedback: str
    iteration_count: int

    # Final Output Layer
    final_analysis: Optional[AnalysisResult]
    final_output: Optional[str]

    # Multi-Turn Conversation Memory Buffer
    messages: Annotated[List[BaseMessage], add_messages]
