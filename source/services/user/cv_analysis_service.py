"""
Unified Candidate CV & Career Analysis Service (Use Case Facade Orchestrator).
Single entry-point for FastAPI Presentation controllers, CLI utilities, and background workers.
Coordinates:
- Deterministic CV Ingestion & Profile extraction (< 5ms)
- O*NET Database Risk Determination (0ms lookup, sub-scores, displacement mechanisms)
- Agglomerative clustering for safe career pivots & transition cost/feasibility analysis
- Skill Countermeasures Matrix (Ma trận Đối sách Kỹ năng) via LLM
- 3-Tier Query Intent Routing (Direct Chat < 200ms, Fast RAG < 800ms, Workflow via ICopilotWorkflow)
- True real-time token streaming via LangChain LLM generator (TTFT < 800ms)
"""

import tempfile
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, Generator, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.risk import RiskProfile, RiskLevel, SubRiskScores
from source.domain.entities.transition import SafeTransitionRole, SkillGap
from source.domain.entities.roadmap import CopilotRoadmap, RoadmapPhase
from source.domain.entities.analysis import AnalysisResult
from source.domain.entities.prompt import CopilotPromptCatalog
from source.domain.interfaces.onet_repository import IOnetRepository
from source.domain.interfaces.copilot import ICopilotWorkflow
from source.domain.interfaces.rag import IEvidenceRetriever
from source.infrastructure.database.repo.onet_repository import OnetRepository
from source.parsers.document_parser import DocumentParser
from source.parsers.deterministic_profile_parser import DeterministicProfileParser
from source.services.user.career_service import UserCareerService
from source.services.user.cv_service import UserCVService
from source.services.user.risk_router import HybridRiskRouter
from source.services.user.query_router import QueryIntentRouter, QueryIntent
from source.model.LLM_model.copilot.nodes import (
    _create_default_reasoning_llm,
    _create_default_fast_llm,
    _create_fallback_roadmap,
)
from source.model.LLM_model.copilot.workflow import LangGraphCopilotWorkflow
from source.model.LLM_model.rag.retriever import HierarchicalEvidenceRetriever
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class CVAnalysisService:
    """
    Unified Application Facade Service.
    Master orchestrator interacting exclusively through domain protocols/interfaces:
    - IOnetRepository
    - ICopilotWorkflow
    - IEvidenceRetriever
    - QueryIntentRouter
    """

    def __init__(
        self,
        onet_repo: Optional[IOnetRepository] = None,
        career_service: Optional[UserCareerService] = None,
        cv_service: Optional[UserCVService] = None,
        risk_router: Optional[HybridRiskRouter] = None,
        copilot_workflow: Optional[ICopilotWorkflow] = None,
        copilot_graph: Optional[Any] = None,  # Legacy alias for test backward-compatibility
        checkpointer: Optional[Any] = None,
        doc_parser: Optional[DocumentParser] = None,
        profile_parser: Optional[DeterministicProfileParser] = None,
        evidence_retriever: Optional[IEvidenceRetriever] = None,
        query_router: Optional[QueryIntentRouter] = None,
    ):
        """
        Initializes the CVAnalysisService with clean dependency injection.
        """
        self._onet_repo = onet_repo
        self._career_service = career_service
        self._cv_service = cv_service
        self._risk_router = risk_router
        self._copilot_workflow = copilot_workflow
        self._copilot_graph = copilot_graph
        self._checkpointer = checkpointer
        self._doc_parser = doc_parser or DocumentParser()
        self._profile_parser = profile_parser or DeterministicProfileParser()
        self._evidence_retriever = evidence_retriever
        self._query_router = query_router or QueryIntentRouter()
        self._sessions: Dict[str, Dict[str, Any]] = {}

        logger.debug(
            "CVAnalysisService initialized (repo=%s, career_service=%s, workflow=%s)",
            type(onet_repo).__name__ if onet_repo else "Default",
            type(career_service).__name__ if career_service else "Default",
            "Injected" if copilot_workflow or copilot_graph else "LazyCompiled",
        )

    @property
    def onet_repo(self) -> IOnetRepository:
        """Lazy-loads OnetRepository from infrastructure."""
        if self._onet_repo is None:
            self._onet_repo = OnetRepository()
        return self._onet_repo

    @property
    def cv_service(self) -> UserCVService:
        """Lazy-loads UserCVService orchestrator."""
        if self._cv_service is None:
            self._cv_service = UserCVService(
                document_parser=self._doc_parser,
                profile_extractor=self._profile_parser,
            )
        return self._cv_service

    @property
    def career_service(self) -> UserCareerService:
        """Lazy-loads UserCareerService orchestrator."""
        if self._career_service is None:
            self._career_service = UserCareerService(onet_repo=self.onet_repo)
        return self._career_service

    @property
    def risk_router(self) -> HybridRiskRouter:
        """Lazy-loads HybridRiskRouter strategy orchestrator."""
        if self._risk_router is None:
            self._risk_router = HybridRiskRouter(
                onet_repo=self.onet_repo,
                risk_predictor=self.career_service.risk_predictor,
            )
        return self._risk_router

    @property
    def evidence_retriever(self) -> IEvidenceRetriever:
        """Lazy-loads HierarchicalEvidenceRetriever."""
        if self._evidence_retriever is None:
            self._evidence_retriever = HierarchicalEvidenceRetriever()
        return self._evidence_retriever

    @property
    def query_router(self) -> QueryIntentRouter:
        """Returns QueryIntentRouter instance."""
        return self._query_router

    @property
    def workflow(self) -> ICopilotWorkflow:
        """Lazy-compiles and returns the ICopilotWorkflow abstraction."""
        if self._copilot_workflow is None:
            if self._copilot_graph is not None:
                # Wrap existing injected graph into workflow adapter
                self._copilot_workflow = LangGraphCopilotWorkflow(
                    graph=self._copilot_graph,
                    checkpointer=self._checkpointer,
                )
            else:
                logger.info("Initializing LangGraphCopilotWorkflow abstraction...")
                self._copilot_workflow = LangGraphCopilotWorkflow(
                    onet_repo=self.onet_repo,
                    career_service=self.career_service,
                    evidence_retriever=self.evidence_retriever,
                    checkpointer=self._checkpointer,
                )
        return self._copilot_workflow

    @property
    def graph(self) -> Any:
        """Legacy property for backward compatibility with tests accessing raw graph."""
        wf = self.workflow
        return getattr(wf, "graph", wf)

    def start_cv_analysis(
        self,
        file_path: Optional[Union[str, Path]] = None,
        raw_cv_text: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Phase 1: Ingests CV document or raw text deterministically (< 5ms),
        preserves raw_text in CandidateProfile, and initiates workflow pausing at HITL checkpoint.
        """
        if not file_path and not raw_cv_text:
            raise ValueError("Either file_path or raw_cv_text must be provided.")

        session = session_id or str(uuid.uuid4())
        start_time = time.perf_counter()

        # 1. Extract raw text & structured profile via UserCVService
        if file_path:
            raw_text = self.cv_service.extract_raw_text(Path(file_path))
        else:
            raw_text = str(raw_cv_text)

        profile = self.cv_service.extract_profile_from_text(raw_text)
        profile.raw_text = raw_text

        # 2. Unified 5-Tier SOC Discovery via IOnetRepository
        res = None
        if hasattr(self.onet_repo, "resolve_candidate_soc"):
            try:
                res = self.onet_repo.resolve_candidate_soc(profile)
            except Exception:
                res = None

        if isinstance(res, tuple) and len(res) == 4:
            soc_code, soc_title, conf, is_student_or_fresher = res
        else:
            raw_title = (profile.current_title or "").strip()
            student_keywords = ["student", "sinh viên", "học sinh", "intern", "thực tập", "fresher", "graduated", "tốt nghiệp"]
            is_student_or_fresher = (
                any(k in raw_title.lower() for k in student_keywords)
                or profile.years_of_experience == 0.0
            )
            soc_code, soc_title, conf = self.onet_repo.match_job_title(raw_title)
            if (not soc_code or conf < 0.60) and profile.get_all_skills and hasattr(self.onet_repo, "match_role_by_skills"):
                s_soc, s_title, s_conf = self.onet_repo.match_role_by_skills(profile.get_all_skills)
                if s_soc and s_conf > conf:
                    soc_code, soc_title, conf = s_soc, s_title, s_conf
            if (not soc_code or conf < 0.50) and profile.education and hasattr(self.onet_repo, "match_role_by_education"):
                e_soc, e_title, e_conf = self.onet_repo.match_role_by_education(profile.education)
                if e_soc and e_conf > conf:
                    soc_code, soc_title, conf = e_soc, e_title, e_conf

        # 3. Delegate to ICopilotWorkflow abstraction
        try:
            workflow_res = self.workflow.start_workflow(
                session_id=session,
                raw_cv_text=raw_text,
                candidate_profile=profile,
            )
            suggested_soc = workflow_res.get("suggested_soc_code") or soc_code
            suggested_title = workflow_res.get("suggested_title") or soc_title
            match_conf = workflow_res.get("match_confidence") if workflow_res.get("match_confidence") is not None else conf
        except Exception as e:
            logger.debug("Workflow start notice: %s. Using direct extracted values.", e)
            suggested_soc = soc_code
            suggested_title = soc_title
            match_conf = conf

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = {
            "session_id": session,
            "status": "AWAITING_HUMAN_REVIEW",
            "candidate_profile": profile,
            "suggested_soc_code": suggested_soc,
            "suggested_title": suggested_title,
            "match_confidence": match_conf,
            "is_fresher_mode": is_student_or_fresher,
            "latency_ms": round(elapsed_ms, 2),
        }

        # Cache session
        self._sessions[session] = {
            "session_id": session,
            "raw_text": raw_text,
            "candidate_profile": profile,
            "suggested_soc_code": suggested_soc,
            "suggested_title": suggested_title,
            "match_confidence": match_conf,
            "is_fresher_mode": is_student_or_fresher,
        }

        logger.info(
            "Phase 1 deterministic extraction completed in %.2f ms. SOC: %s, Conf: %.3f, FresherMode: %s",
            elapsed_ms, suggested_soc, match_conf, is_student_or_fresher
        )
        return result

    def confirm_and_resume_analysis(
        self,
        session_id: str,
        user_confirmed: bool = True,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Phase 2: Confirms candidate occupation, executes O*NET database risk lookup,
        clustering Top 3 safe pivots, RAG evidence grounding, LLM countermeasures, and compiles AnalysisResult.
        """
        start_time = time.perf_counter()
        session_data = self._sessions.get(session_id)

        if not session_data and not self.workflow.get_session_state(session_id):
            raise KeyError(f"No active session found for session_id '{session_id}'.")

        if not session_data:
            snapshot = self.workflow.get_session_state(session_id) or {}
            session_data = dict(snapshot)

        # 1. Attempt full resumption through LangGraph Copilot Workflow (StateGraph)
        try:
            analysis = self.workflow.resume_workflow(
                session_id=session_id,
                user_confirmed=user_confirmed,
                overrides=overrides,
            )
            if analysis and (not analysis.session_id or analysis.session_id != session_id):
                analysis.session_id = session_id
            session_data["final_analysis"] = analysis
            session_data["candidate_profile"] = analysis.candidate_profile
            self._sessions[session_id] = session_data
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info("Workflow resumed successfully through LangGraph in %.2f ms. AnalysisResult: %s", elapsed_ms, analysis.session_id)
            return analysis
        except Exception as e:
            logger.warning("Workflow resume through LangGraph failed (%s). Falling back to dynamic role-aligned synthesis.", e)

        # Apply overrides to local profile if present
        profile: CandidateProfile = session_data.get("candidate_profile") or CandidateProfile()
        if overrides:
            if overrides.get("candidate_name"):
                profile.full_name = overrides["candidate_name"]
            if overrides.get("current_occupation"):
                profile.current_title = overrides["current_occupation"]
            if overrides.get("years_of_experience") is not None:
                profile.years_of_experience = float(overrides["years_of_experience"])
            if overrides.get("technical_skills"):
                profile.tools_and_technologies = overrides["technical_skills"]

        soc_code = (overrides.get("matched_soc_code") if overrides else None) or session_data.get("suggested_soc_code")

        # 1. Authoritative Risk Profile via HybridRiskRouter
        risk = self.risk_router.determine_risk(
            profile=profile,
            soc_code=soc_code,
            confidence=session_data.get("match_confidence", 0.0),
            user_confirmed=user_confirmed,
        )

        # 2. Safe Career Pivots (Top 3) & Cost Feasibility
        pivots = []
        if soc_code:
            pivots = self.career_service.get_safe_career_pivots(
                soc_code=soc_code,
                current_threat=risk.threat_score,
                top_k=3,
            )
        primary_pivot = pivots[0] if pivots else None

        # 3. Build SkillGaps for all Top 2-3 pivots
        candidate_skills = set(profile.get_all_skills)
        skill_gaps: List[SkillGap] = []
        for p in pivots[:3]:
            vulnerable = [s for s in candidate_skills if s not in p.matched_skills and s not in p.defense_skills]
            gap = SkillGap(
                target_role_title=p.target_title,
                target_soc_code=p.target_soc_code,
                strengths=p.matched_skills,
                vulnerable_skills=vulnerable,
                defense_skills=p.defense_skills,
                missing_skills=p.missing_critical_skills,
                matched_tools=p.matched_tools,
                missing_critical_tools=p.missing_critical_tools,
                hot_technologies_to_learn=p.hot_technologies_to_learn,
                overlap_percentage=p.similarity_pct,
            )
            skill_gaps.append(gap)

        primary_gap = skill_gaps[0] if skill_gaps else None
        target_title = primary_pivot.target_title if primary_pivot else (profile.current_title or "Professional Role")
        target_soc = primary_pivot.target_soc_code if primary_pivot else (soc_code or "15-1252.00")

        roadmap = _create_fallback_roadmap(target_title, target_soc, primary_gap)

        if primary_gap:
            primary_tool = primary_gap.missing_critical_tools[0] if primary_gap.missing_critical_tools else "công cụ chuyên môn"
            cert = roadmap.phases[0].recommended_certifications[0] if roadmap.phases and roadmap.phases[0].recommended_certifications else "Chứng chỉ Quốc tế"
            primary_gap.countermeasures = [
                f"Tận dụng kinh nghiệm {profile.current_title or 'chuyên môn'} để học nhanh {primary_tool} qua các bài lab thực chiến",
                f"Xây dựng dự án Portfolio mã nguồn mở chứng minh năng lực {target_title}",
                f"Đầu tư ôn luyện chứng chỉ {cert} để chuẩn hóa hồ sơ năng lực cạnh tranh trước AI",
            ]

        analysis = AnalysisResult(
            session_id=session_id,
            candidate_profile=profile,
            risk_profile=risk,
            safe_pivots=pivots,
            skill_gap=primary_gap,
            skill_gaps=skill_gaps,
            roadmap=roadmap,
        )

        session_data["final_analysis"] = analysis
        session_data["candidate_profile"] = profile
        self._sessions[session_id] = session_data

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Phase 2 analysis completed in %.2f ms. Target role: '%s'", elapsed_ms, roadmap.target_role)
        return analysis

    def chat_with_copilot(
        self,
        session_id: str,
        message: str,
    ) -> str:
        """
        Synchronous chat response generator.
        """
        if not message or not message.strip():
            raise ValueError("Chat message must not be empty.")
        chunks = list(self.stream_chat_with_copilot(session_id=session_id, message=message))
        return "".join(chunks)

    def _get_history_messages(self, session_data: Dict[str, Any], max_turns: int = 6) -> List[Any]:
        """Extracts recent conversation turns formatted for LangChain LLM input."""
        history = session_data.get("chat_history", [])
        messages: List[Any] = []
        for turn in history[-max_turns:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                # Strip the execution badge from history so LLM doesn't learn or repeat old timestamps
                clean_content = content.split("\n\n---\n⏱️")[0].strip()
                if clean_content:
                    messages.append(AIMessage(content=clean_content))
        return messages

    def _append_to_history(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Appends user prompt and assistant response to session memory."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {}
        history = self._sessions[session_id].setdefault("chat_history", [])
        if user_msg:
            history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            history.append({"role": "assistant", "content": assistant_msg})
        # Maintain rolling window of last 12 turns max to prevent memory bloat
        if len(history) > 12:
            self._sessions[session_id]["chat_history"] = history[-12:]

    def stream_chat_with_copilot(
        self,
        session_id: str,
        message: str = "",
        file_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        True Real-Time SSE Token Streamer with 3-Tier Query Dispatcher and Multi-turn Memory:
        - Tier 1 (DIRECT_CHAT): Fast conversational greetings & guidance (< 200ms).
        - Tier 2 (KNOWLEDGE_RAG): Factual O*NET benchmark Q&A (< 800ms) without LangGraph overhead.
        - Tier 3 (WORKFLOW): Full CV ingestion, Skill Countermeasures Matrix, and Stateful Workflow.
        """
        session = session_id or str(uuid.uuid4())
        session_data = self._sessions.setdefault(session, {})
        has_file = file_bytes is not None and len(file_bytes) > 0
        has_cv = "final_analysis" in session_data or "candidate_profile" in session_data

        # Dispatch Query Intent
        intent = self.query_router.route(message=message, has_file=has_file, session_has_cv=has_cv)
        logger.info(
            "Dispatching interaction for session '%s' -> Intent: %s (has_file=%s, has_cv=%s)",
            session, intent.value, has_file, has_cv
        )

        def _dispatch_generator():
            if has_file:
                yield from self._stream_cv_analysis(session, file_bytes, filename or "resume.pdf", message)
            elif intent == QueryIntent.DIRECT_CHAT:
                yield from self._stream_direct_chat(session, message, session_data, has_cv=has_cv)
            elif intent == QueryIntent.KNOWLEDGE_RAG:
                yield from self._stream_knowledge_rag(session, message, session_data, has_cv=has_cv)
            else:
                yield from self._stream_workflow_advice(session, message, session_data, has_cv=has_cv)

        accumulated_response = []
        for chunk in _dispatch_generator():
            accumulated_response.append(chunk)
            yield chunk

        # Save complete turn to session memory
        full_response_text = "".join(accumulated_response)
        self._append_to_history(session, message, full_response_text)

    def _stream_direct_chat(
        self,
        session_id: str,
        message: str,
        session_data: Dict[str, Any],
        has_cv: bool = False,
    ) -> Generator[str, None, None]:
        """
        Tier 1: Direct Chat / Greetings / Help (< 200ms).
        """
        start_t = time.perf_counter()
        msg_lower = message.lower().strip()
        if any(w in msg_lower for w in ["alo", "alô", "hi", "hello", "chào", "xin chào", "hey"]):
            yield CopilotPromptCatalog.build_direct_chat_greeting(has_cv=has_cv)
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            yield f"\n\n---\n⏱️ **Thời gian phản hồi:** {elapsed_ms:.1f}ms | 🧭 **Luồng xử lý:** DIRECT_CHAT"
            return

        if any(w in msg_lower for w in ["bạn là ai", "bạn tên gì", "giúp gì", "hướng dẫn", "tính năng"]):
            yield CopilotPromptCatalog.build_direct_chat_help(has_cv=has_cv)
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            yield f"\n\n---\n⏱️ **Thời gian phản hồi:** {elapsed_ms:.1f}ms | 🧭 **Luồng xử lý:** DIRECT_CHAT"
            return

        # Fallback to Fast LLM direct with history
        cta_instruction = (
            "Tuyệt đối KHÔNG nhắc người dùng tải lên CV vì họ đã tải lên CV trong phiên này."
            if has_cv
            else "Nhắc nhẹ người dùng có thể nhấn nút chiếc ghim (📎) để tải lên CV phân tích miễn phí."
        )
        system_prompt = f"""Bạn là AI Career Copilot. Hãy trả lời câu hỏi của người dùng một cách thân thiện, ngắn gọn và hữu ích bằng tiếng Việt.
{cta_instruction}"""
        llm = _create_default_fast_llm()
        history_msgs = self._get_history_messages(session_data, max_turns=6)
        messages = [SystemMessage(content=system_prompt)] + history_msgs + [HumanMessage(content=message)]
        try:
            for chunk in llm.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    yield token
        except Exception as e:
            logger.warning("Direct chat LLM stream fallback: %s", e)
            if has_cv:
                yield f"Tôi đã ghi nhận tin nhắn của bạn: *\"{message}\"*. Bạn có câu hỏi nào cần làm rõ về lộ trình chuyển đổi hay các kỹ năng phòng thủ không?"
            else:
                yield f"Tôi đã ghi nhận tin nhắn của bạn: *\"{message}\"*. Bạn hãy nhấn nút chiếc ghim (📎) bên dưới để gửi CV phân tích chuyên sâu nhé!"

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        yield f"\n\n---\n⏱️ **Thời gian phản hồi:** {elapsed_ms:.1f}ms | 🧭 **Luồng xử lý:** DIRECT_CHAT"

    def _stream_knowledge_rag(
        self,
        session_id: str,
        message: str,
        session_data: Dict[str, Any],
        has_cv: bool = False,
    ) -> Generator[str, None, None]:
        """
        Tier 2: Fast Knowledge RAG without LangGraph state machine.
        Retrieves factual evidence from O*NET and streams grounded answer in < 800ms.
        """
        start_t = time.perf_counter()
        logger.info("Executing Fast Knowledge RAG for query: '%s' (has_cv=%s)...", message[:60], has_cv)
        
        # Scoped retrieval if candidate SOC code is available in session
        soc_code = session_data.get("suggested_soc_code")
        try:
            chunks = self.evidence_retriever.retrieve_evidence(query=message, soc_code=soc_code, top_k=settings.RAG_TOP_K)
        except Exception as e:
            logger.warning("RAG retrieval fallback: %s", e)
            chunks = []

        # Candidate context summary if available
        candidate_summary = ""
        profile: Optional[CandidateProfile] = session_data.get("candidate_profile")
        risk: Optional[RiskProfile] = session_data.get("risk_profile")
        analysis: Optional[AnalysisResult] = session_data.get("final_analysis")
        if profile:
            candidate_summary = (
                f"- Ứng viên: {profile.full_name} ({profile.current_title or 'Sinh viên / Fresher'})\n"
                f"- Kinh nghiệm: {profile.years_of_experience:.1f} năm\n"
                f"- Kỹ năng nhận diện: {', '.join(profile.tools_and_technologies[:6])}\n"
            )
            if risk:
                candidate_summary += f"- Điểm đe dọa AI: {risk.threat_score:.1f}% ({risk.risk_level})\n"
            if analysis and analysis.safe_pivots:
                p_names = [p.target_title for p in analysis.safe_pivots[:2]]
                candidate_summary += f"- Các ngành an toàn đề xuất: {', '.join(p_names)}\n"

        system_prompt = CopilotPromptCatalog.build_knowledge_rag_prompt(
            query=message,
            evidences=chunks,
            has_cv=has_cv,
            candidate_summary=candidate_summary,
        )

        llm = _create_default_fast_llm()
        history_msgs = self._get_history_messages(session_data, max_turns=6)
        messages = [SystemMessage(content=system_prompt)] + history_msgs + [HumanMessage(content=message)]

        try:
            for chunk in llm.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    yield token
        except Exception as e:
            logger.warning("Fast RAG LLM stream fallback: %s", e)
            evidence_summary = "\n".join([f"- [{c.get('soc_code', 'O*NET')}]: {c.get('content', '')[:300]}" for c in chunks])
            fallback_cta = "" if has_cv else "\n\n👉 Bạn hãy nhấn vào nút chiếc ghim (📎) bên dưới để tải lên CV và nhận phân tích cá nhân hóa nhé!"
            yield f"Dựa trên CSDL O*NET:\n{evidence_summary}{fallback_cta}"

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        yield f"\n\n---\n⏱️ **Thời gian phản hồi:** {elapsed_ms:.1f}ms | 🧭 **Luồng xử lý:** KNOWLEDGE_RAG"

    def _stream_workflow_advice(
        self,
        session_id: str,
        message: str,
        session_data: Dict[str, Any],
        has_cv: bool = False,
    ) -> Generator[str, None, None]:
        """
        Tier 3: Stateful Workflow Chat with Context Memory and Multi-Pivot Explainability.
        """
        start_t = time.perf_counter()
        analysis: Optional[AnalysisResult] = session_data.get("final_analysis")
        profile: Optional[CandidateProfile] = session_data.get("candidate_profile")

        if analysis and analysis.roadmap:
            target_role = analysis.roadmap.target_role
            pivots_desc = ""
            if analysis.safe_pivots:
                p_lines = []
                for p in analysis.safe_pivots[:3]:
                    meta = self.onet_repo.get_metadata(p.target_soc) if hasattr(self.onet_repo, "get_metadata") else None
                    salary_str = f", Lương trung vị O*NET: ${meta.get('median_salary', 0):,.0f}/năm" if meta and meta.get("median_salary") else ""
                    insight_str = f" | Ghi chú: {meta.get('insight', '')[:100]}..." if meta and meta.get("insight") else ""
                    p_lines.append(
                        f"   + {p.target_title} (SOC: {p.target_soc}, Độ tương đồng: {p.similarity_pct:.1f}%, Giảm rủi ro AI: {p.threat_reduction:.1f}%{salary_str}, Thiếu công cụ: {', '.join(p.missing_critical_tools[:3])}{insight_str})"
                    )
                pivots_desc = "\nTop các hướng chuyển dịch an toàn đã đề xuất:\n" + "\n".join(p_lines)

            # Scoped Dynamic Re-RAG Retrieval
            evidence_text = ""
            try:
                target_soc = analysis.safe_pivots[0].target_soc if analysis.safe_pivots else None
                rag_chunks = self.evidence_retriever.retrieve_evidence(
                    query=message,
                    soc_code=target_soc,
                    top_k=3,
                )
                if rag_chunks:
                    evidence_lines = [
                        f"- [{c.get('soc_code', 'O*NET')}]: {c.get('content', '')[:250]}"
                        for c in rag_chunks
                    ]
                    evidence_text = "\n\nDữ liệu thực chứng O*NET liên quan đến câu hỏi:\n" + "\n".join(evidence_lines)
            except Exception as e:
                logger.debug("Workflow Re-RAG retrieval skipped or failed: %s", e)

            system_prompt = f"""Bạn là AI Career Copilot - Chuyên gia Cố vấn Chiến lược Nghề nghiệp cao cấp.
Bạn đang trò chuyện và cố vấn trực tiếp cho ứng viên: {analysis.candidate_profile.full_name} ({analysis.candidate_profile.current_title or 'Sinh viên / Fresher'}).
- Điểm nguy cơ tự động hóa AI: {analysis.risk_profile.threat_score:.1f}% ({analysis.risk_profile.risk_level}).
- Tác vụ lặp lại (sub_task_automation): {analysis.risk_profile.sub_scores.task_automation:.1f}%.
- Độ hoàn thiện công cụ AI (sub_tool_maturity): {analysis.risk_profile.sub_scores.tool_maturity:.1f}%.
- Phơi nhiễm Agentic (sub_agentic_exposure): {analysis.risk_profile.sub_scores.agentic_exposure:.1f}%.
- Kỹ năng phòng thủ: {', '.join(analysis.risk_profile.defense_skills)}.
- Vai trò mục tiêu đề xuất: {target_role}.
{pivots_desc}
{evidence_text}

NGUYÊN TẮC HỘI THOẠI & ĐỊNH DẠNG:
1. Trả lời trực diện vào câu hỏi của ứng viên bằng tiếng Việt, bám sát các số liệu O*NET và phân tích hồ sơ ở trên.
2. Khi so sánh giữa các nghề nghiệp, BẮT BUỘC sử dụng Markdown Table (| Tiêu chí | Nghề 1 | Nghề 2 |) với các tiêu chí: Chuyên môn, Kỹ năng cần bù đắp, Độ khó tiếp cận, Mức lương thị trường (USD/năm).
3. Khi phân tích về "Cơ hội phát triển", BẮT BUỘC trình bày đầy đủ:
   - Lộ trình thăng tiến từng bước: Junior -> Mid -> Senior -> Lead/Architect.
   - Nhu cầu thị trường & triển vọng trước làn sóng AI hóa.
   - Các ngách chuyên môn giá trị cao.
4. Tuyệt đối KHÔNG viết ký tự '#' cô lập trên một dòng. Khi dùng danh sách gạch đầu dòng, viết nội dung giải thích đầy đủ, không bỏ lửng tiêu đề mục.
5. Giải quyết chính xác các đại từ tham chiếu ('hướng thứ nhất', 'hai lựa chọn trên', 'điểm của tôi') dựa trên lịch sử hội thoại.
6. Tuyệt đối KHÔNG nhắc người dùng tải lên CV vì họ ĐÃ tải lên CV trong phiên này.
7. Trả lời súc tích, hoàn chỉnh, chuyên nghiệp (khoảng 250-450 từ), không bao giờ cắt cụt câu giữa chừng."""
        else:
            yield from self._stream_direct_chat(session_id, message, session_data, has_cv=has_cv)
            return

        llm = _create_default_fast_llm()
        history_msgs = self._get_history_messages(session_data, max_turns=6)
        messages = [SystemMessage(content=system_prompt)] + history_msgs + [HumanMessage(content=message or "Xin chào!")]

        try:
            for chunk in llm.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    yield token
        except Exception as e:
            logger.warning("Workflow LLM stream fallback: %s", e)
            if analysis and analysis.roadmap and len(analysis.roadmap.phases) > 0:
                p1 = analysis.roadmap.phases[0]
                cert = p1.recommended_certifications[0] if p1.recommended_certifications else "nền tảng"
                yield f"Dựa trên lộ trình 90 ngày của bạn hướng tới {analysis.roadmap.target_role}: Đối với câu hỏi \"{message}\", ở {p1.phase_name}, mục tiêu trọng tâm là {p1.objective}. Bạn nên ưu tiên hoàn thành chứng chỉ {cert} và các milestone đề ra trong kế hoạch."
            else:
                yield f"Tôi đã ghi nhận câu hỏi của bạn: *\"{message}\"*. Hãy tập trung vào việc bổ sung các kỹ năng phòng thủ và hoàn thiện dự án thực chiến nhé!"

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        yield f"\n\n---\n⏱️ **Thời gian phản hồi:** {elapsed_ms:.1f}ms | 🧭 **Luồng xử lý:** WORKFLOW (Tư vấn Lộ trình & Đối sách Kỹ năng)"


    def _stream_cv_analysis(
        self,
        session_id: str,
        file_bytes: bytes,
        filename: str,
        user_message: str = "",
    ) -> Generator[str, None, None]:
        """
        Executes end-to-end CV analysis through the unified ICopilotWorkflow / StateGraph pipeline
        and streams the advisory report token-by-token.
        Eliminates duplicate shadow pipelines by leveraging start_cv_analysis and confirm_and_resume_analysis.
        """
        start_t = time.perf_counter()
        suffix = Path(filename).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            # 1. Phase 1: Ingest & Extract profile deterministically (< 5ms)
            init_res = self.start_cv_analysis(file_path=tmp_path, session_id=session_id)
            profile: CandidateProfile = init_res["candidate_profile"]
            soc_code = init_res.get("suggested_soc_code")
            soc_title = init_res.get("suggested_title")
            is_student_or_fresher = init_res.get("is_fresher_mode", False)

            # If completely no role detected (empty CV)
            if not soc_code:
                yield CopilotPromptCatalog.build_no_role_detected_response()
                elapsed_ms = (time.perf_counter() - start_t) * 1000
                yield f"\n\n---\n⏱️ **Thời gian phản hồi:** {elapsed_ms:.1f}ms | 🧭 **Luồng xử lý:** WORKFLOW (Phân tích CV)"
                return

            # 2. Phase 2: Resume workflow through StateGraph (RAG Grounding, Copilot Roadmap, Critic Evaluator)
            analysis = self.confirm_and_resume_analysis(
                session_id=session_id,
                user_confirmed=True,
            )

            # 3. Stream Advisory Response using Prompt Catalog & Reasoning LLM
            pivots = analysis.safe_pivots or []
            skill_gaps = analysis.skill_gaps or ([analysis.skill_gap] if analysis.skill_gap else [])
            risk = analysis.risk_profile

            if is_student_or_fresher:
                system_prompt = CopilotPromptCatalog.build_student_onboarding_prompt(
                    candidate=profile,
                    safe_pivots=pivots,
                    skill_gaps=skill_gaps,
                )
            else:
                system_prompt = CopilotPromptCatalog.build_multi_pivot_advisory_prompt(
                    candidate=profile,
                    risk_profile=risk,
                    safe_pivots=pivots,
                    skill_gaps=skill_gaps,
                    grounding_evidences=analysis.grounding_evidences or [],
                )

            llm = _create_default_reasoning_llm()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message or "Hãy phân tích chi tiết hồ sơ CV của tôi."),
            ]

            for chunk in llm.stream(messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    yield token

        except Exception as e:
            logger.exception("Error during unified CV analysis streaming: %s", e)
            # Structured fallback based on actual analysis result if available
            session_data = self._sessions.get(session_id, {})
            analysis = session_data.get("final_analysis")
            if analysis and analysis.risk_profile:
                yield f"### ⚠️ NGUY CƠ TỰ ĐỘNG HÓA AI: {analysis.risk_profile.threat_score:.1f}% ({analysis.risk_profile.risk_level})\n\n"
                if analysis.safe_pivots:
                    yield "### 🔄 CÁC HƯỚNG CHUYỂN DỊCH AN TOÀN ĐỀ XUẤT:\n"
                    for idx, p in enumerate(analysis.safe_pivots[:3], 1):
                        yield f"{idx}. **{p.target_title}** — Độ tương đồng: {p.similarity_pct:.1f}%, Giảm rủi ro AI: {p.threat_reduction:.1f}%\n"
                if analysis.roadmap:
                    yield f"\n### 📅 LỘ TRÌNH 90 NGÀY:\n{analysis.roadmap.executive_summary}\n"
            else:
                yield f"Đã xảy ra lỗi trong quá trình phân tích CV: {e}. Vui lòng thử lại hoặc tải lên CV dạng văn bản rõ ràng hơn."
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

            elapsed_ms = (time.perf_counter() - start_t) * 1000
            yield f"\n\n---\n⏱️ **Thời gian phản hồi:** {elapsed_ms:.1f}ms | 🧭 **Luồng xử lý:** WORKFLOW (Phân tích CV Chuyên sâu qua LangGraph)"

    def get_analysis_result(self, session_id: str) -> Optional[AnalysisResult]:
        """
        Retrieves analysis result from memory cache or ICopilotWorkflow state.
        """
        session_data = self._sessions.get(session_id, {})
        if "final_analysis" in session_data:
            return session_data["final_analysis"]

        state = self.workflow.get_session_state(session_id)
        if state and "final_analysis" in state:
            return state["final_analysis"]
        return None


# Global Singleton Factory for FastAPI Dependency Injection
_global_cv_analysis_service: Optional[CVAnalysisService] = None


def get_cv_analysis_service() -> CVAnalysisService:
    """FastAPI Dependency Provider singleton returning CVAnalysisService."""
    global _global_cv_analysis_service
    if _global_cv_analysis_service is None:
        _global_cv_analysis_service = CVAnalysisService()
    return _global_cv_analysis_service
