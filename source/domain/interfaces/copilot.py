"""
Domain Protocol for Copilot Agent Workflow.
Defines abstract interface for orchestrating the AI Career Copilot execution pipeline,
decoupling Application Services from underlying LangGraph / State Machine implementations.
"""

from typing import Protocol, runtime_checkable, Optional, Dict, Any, Generator
from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.analysis import AnalysisResult


@runtime_checkable
class ICopilotWorkflow(Protocol):
    """
    Domain abstraction for multi-step career advisory workflows.
    Encapsulates state graph compilation, memory checkpointers,
    HITL interruptions, and reasoning loops.
    """

    def start_workflow(
        self,
        session_id: str,
        raw_cv_text: str,
        candidate_profile: Optional[CandidateProfile] = None,
    ) -> Dict[str, Any]:
        """
        Initiates Phase 1 CV ingestion and pauses at the Human-in-the-Loop checkpoint.

        Args:
            session_id: Unique thread identifier
            raw_cv_text: Raw textual resume content
            candidate_profile: Pre-extracted candidate profile (optional)

        Returns:
            Dict containing status, candidate_profile, suggested_soc_code, suggested_title, match_confidence.
        """
        ...

    def resume_workflow(
        self,
        session_id: str,
        user_confirmed: bool = True,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Resumes execution from Phase 2 (post-HITL), computes risk, safe pivots,
        RAG evidence grounding, LLM skill countermeasures matrix, and compiles AnalysisResult.

        Args:
            session_id: Active session identifier
            user_confirmed: Whether candidate approved extracted profile
            overrides: Optional human corrections (e.g. matched_title, soc_code)

        Returns:
            AnalysisResult domain aggregate
        """
        ...

    def stream_workflow(
        self,
        session_id: str,
        message: str = "",
        candidate_profile: Optional[CandidateProfile] = None,
    ) -> Generator[str, None, None]:
        """
        Streams conversational token-by-token advisory responses with evidence grounding.

        Args:
            session_id: Thread/session identifier
            message: User follow-up question or instruction
            candidate_profile: Optional profile context

        Yields:
            Token chunks as strings
        """
        ...

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves active workflow state snapshot from memory checkpointer.

        Args:
            session_id: Session identifier

        Returns:
            State dictionary or None if not found
        """
        ...
