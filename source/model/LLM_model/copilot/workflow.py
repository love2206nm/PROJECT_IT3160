"""
LangGraph Copilot Workflow Adapter.
Implements ICopilotWorkflow domain protocol to encapsulate the LangGraph StateGraph,
memory checkpointing, HITL interruptions, and reasoning loops behind a clean abstraction.
"""

import time
import uuid
from typing import Dict, Any, Optional, Generator

from langgraph.checkpoint.memory import MemorySaver

from source.domain.interfaces.copilot import ICopilotWorkflow
from source.domain.interfaces.onet_repository import IOnetRepository
from source.domain.interfaces.parser import IDocumentParser, IProfileExtractor
from source.domain.interfaces.rag import IEvidenceRetriever
from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.analysis import AnalysisResult
from source.model.LLM_model.copilot.graph import build_copilot_graph
from source.model.LLM_model.copilot.nodes import _create_default_reasoning_llm
from source.infrastructure.logger import get_logger
from langchain_core.messages import SystemMessage, HumanMessage

logger = get_logger(__name__)


class LangGraphCopilotWorkflow(ICopilotWorkflow):
    """
    Adapter implementing ICopilotWorkflow protocol.
    Decouples Application Services (e.g. CVAnalysisService) from LangGraph StateGraph internals.
    """

    def __init__(
        self,
        doc_parser: Optional[IDocumentParser] = None,
        profile_parser: Optional[IProfileExtractor] = None,
        onet_repo: Optional[IOnetRepository] = None,
        career_service: Optional[Any] = None,
        evidence_retriever: Optional[IEvidenceRetriever] = None,
        copilot_llm: Optional[Any] = None,
        critic_llm: Optional[Any] = None,
        checkpointer: Optional[Any] = None,
        interrupt_before: Optional[list] = None,
        graph: Optional[Any] = None,
    ) -> None:
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = graph or build_copilot_graph(
            doc_parser=doc_parser,
            profile_parser=profile_parser,
            onet_repo=onet_repo,
            career_service=career_service,
            evidence_retriever=evidence_retriever,
            copilot_llm=copilot_llm,
            critic_llm=critic_llm,
            checkpointer=self.checkpointer,
            interrupt_before=interrupt_before if interrupt_before is not None else ["human_review_node"],
        )
        logger.info("LangGraphCopilotWorkflow initialized and compiled successfully.")

    def start_workflow(
        self,
        session_id: str,
        raw_cv_text: str,
        candidate_profile: Optional[CandidateProfile] = None,
    ) -> Dict[str, Any]:
        """
        Executes Phase 1 CV Ingestion and extraction, pausing before Human-in-the-Loop review.
        """
        start_time = time.perf_counter()
        session = session_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": session}}

        initial_state: Dict[str, Any] = {
            "session_id": session,
            "raw_cv_text": raw_cv_text,
            "iteration_count": 0,
        }
        if candidate_profile:
            initial_state["candidate_profile"] = candidate_profile

        # Run through CV ingestion to HITL interrupt
        for _ in self.graph.stream(initial_state, config=config):
            pass

        # Retrieve state from checkpointer
        snapshot = self.graph.get_state(config)
        state_values = snapshot.values if snapshot else {}

        profile = state_values.get("candidate_profile") or candidate_profile
        suggested_soc = state_values.get("matched_soc_code")
        suggested_title = state_values.get("matched_title")
        match_confidence = state_values.get("match_confidence", 0.0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Workflow Phase 1 completed in %.2f ms for session '%s' (SOC: %s, Conf: %.3f)",
            elapsed_ms, session, suggested_soc, match_confidence
        )

        return {
            "session_id": session,
            "status": "AWAITING_HUMAN_REVIEW",
            "candidate_profile": profile,
            "suggested_soc_code": suggested_soc,
            "suggested_title": suggested_title,
            "match_confidence": match_confidence,
            "latency_ms": round(elapsed_ms, 2),
        }

    def resume_workflow(
        self,
        session_id: str,
        user_confirmed: bool = True,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Resumes Phase 2 execution with human feedback/overrides, proceeding through
        risk scoring, pivot discovery, RAG evidence grounding, Copilot roadmap, and Critic reflection.
        """
        start_time = time.perf_counter()
        config = {"configurable": {"thread_id": session_id}}

        # Update state with human confirmation and overrides
        update_payload: Dict[str, Any] = {
            "user_confirmed": user_confirmed,
            "session_id": session_id,
        }
        if overrides:
            if overrides.get("matched_title"):
                update_payload["matched_title"] = overrides["matched_title"]
            if overrides.get("matched_soc_code"):
                update_payload["matched_soc_code"] = overrides["matched_soc_code"]

        self.graph.update_state(config, update_payload, as_node="human_review_node")

        # Resume execution until END
        for _ in self.graph.stream(None, config=config):
            pass

        snapshot = self.graph.get_state(config)
        final_analysis = snapshot.values.get("final_analysis") if snapshot else None

        if final_analysis and isinstance(final_analysis, AnalysisResult):
            if not final_analysis.session_id or final_analysis.session_id != session_id:
                final_analysis.session_id = session_id
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info("Workflow Phase 2 resumed successfully in %.2f ms for session '%s'.", elapsed_ms, session_id)
            return final_analysis

        raise RuntimeError(f"Workflow execution completed but failed to produce AnalysisResult for session '{session_id}'.")

    def stream_workflow(
        self,
        session_id: str,
        message: str = "",
        candidate_profile: Optional[CandidateProfile] = None,
    ) -> Generator[str, None, None]:
        """
        Streams conversational advice token-by-token anchored in workflow state.
        """
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self.graph.get_state(config)
        state_values = snapshot.values if snapshot else {}

        profile = candidate_profile or state_values.get("candidate_profile")
        skill_gap = state_values.get("skill_gap")
        roadmap = state_values.get("copilot_roadmap")

        target_role = skill_gap.target_role_title if skill_gap else (roadmap.target_role if roadmap else "Chuyên viên Công nghệ")
        candidate_name = profile.full_name if profile else "Ứng viên"

        system_prompt = f"""Bạn là AI Career Copilot - Chuyên gia Cố vấn Chiến lược Nghề nghiệp cao cấp.
Bạn đang trò chuyện với: {candidate_name}.
Vai trò mục tiêu đề xuất: {target_role}.
Lộ trình định hướng: {roadmap.executive_summary if roadmap else 'Lộ trình chuyển đổi an toàn thích ứng AI'}.

Hãy trả lời câu hỏi trực tiếp, chuyên nghiệp bằng tiếng Việt, sử dụng Markdown rõ ràng."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message or "Xin chào!"),
        ]

        llm = _create_default_reasoning_llm()
        try:
            for chunk in llm.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    yield token
        except Exception as e:
            logger.warning("Workflow stream fallback on error: %s", e)
            if roadmap and len(roadmap.phases) > 0:
                p1 = roadmap.phases[0]
                cert = p1.recommended_certifications[0] if p1.recommended_certifications else "nền tảng"
                yield f"Dựa trên lộ trình 90 ngày của bạn hướng tới {target_role}: Ở {p1.phase_name}, mục tiêu trọng tâm là {p1.objective}. Bạn nên ưu tiên hoàn thành chứng chỉ {cert}."
            else:
                yield f"Tôi đã ghi nhận câu hỏi của bạn về lộ trình {target_role}. Hãy tập trung nâng cao các kỹ năng phòng thủ AI và hoàn thiện portfolio nhé!"

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves snapshot values for the given session thread.
        """
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self.graph.get_state(config)
        return snapshot.values if snapshot else None
