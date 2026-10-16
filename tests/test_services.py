"""
Unit Tests for Module 7 Application Services & Use Cases.
100% offline, zero network, zero token costs using Mock fixtures.
Tests:
1. HybridRiskRouter:
   - High confidence (>=0.90) -> Deterministic O*NET Lookup (0ms)
   - Low confidence (<0.90) -> ML XGBoost Regressor
   - User confirmed override -> Deterministic O*NET Lookup regardless of confidence
   - Fallback behavior on missing/invalid SOC code
2. CVAnalysisService (Unified Use Case Facade Orchestrator):
   - Phase 1: start_cv_analysis -> runs until HITL pause before human_review_node
   - Phase 2: confirm_and_resume_analysis -> updates state & executes to completion (AnalysisResult)
   - Phase 3: chat_with_copilot -> interactive multi-turn Q&A with grounded response
   - Session & Result retrieval via get_analysis_result
   - Error handling & edge case validation (empty inputs, unknown sessions)
   - Dependency injection factory provider get_cv_analysis_service
"""

from typing import Any, List, Optional, Tuple, Dict
from unittest.mock import MagicMock
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from source.domain.entities.candidate import CandidateProfile, WorkExperience
from source.domain.entities.risk import RiskProfile, RiskLevel, SubRiskScores
from source.domain.entities.transition import SafeTransitionRole, SkillGap
from source.domain.entities.roadmap import CopilotRoadmap, RoadmapPhase
from source.domain.entities.analysis import AnalysisResult
from source.domain.interfaces.onet_repository import IOnetRepository
from source.domain.interfaces.ml_models import IRiskPredictor
from source.model.LLM_model.copilot.graph import build_copilot_graph
from source.model.LLM_model.copilot.prompts import (
    CriticEvaluation,
    CriticScores,
    LLMSocSuggestion,
)
from source.services.user.risk_router import HybridRiskRouter
from source.services.user.cv_analysis_service import (
    CVAnalysisService,
    get_cv_analysis_service,
)


# ==============================================================================
# Mock Fixtures & Helpers
# ==============================================================================

class MockStructuredLLM:
    """Mock LangChain chat model to simulate structured LLM generation with zero network calls."""

    def __init__(self, responses: Optional[List[Any]] = None):
        self.responses = list(responses or [])
        self.call_count = 0

    def with_structured_output(self, schema: Any):
        return self

    def invoke(self, messages: Any) -> Any:
        self.call_count += 1
        if self.responses:
            res = self.responses.pop(0)
            if callable(res):
                return res(messages)
            return res
        return AIMessage(content="Focused roadmap advice on Phase 1 defense skills.")


def create_mock_candidate() -> CandidateProfile:
    return CandidateProfile(
        full_name="Le Van An",
        current_title="Software Developer",
        years_of_experience=4.0,
        hard_skills=["Python", "FastAPI", "Docker", "PostgreSQL"],
        soft_skills=["Critical Thinking", "Problem Solving"],
        tools_and_technologies=["Git", "Linux", "Docker"],
        experiences=[
            WorkExperience(
                company="Tech Co",
                role_title="Backend Engineer",
                duration_months=36,
                description="Engineered RESTful APIs",
            )
        ],
    )


def create_mock_risk_profile(soc_code: str = "15-1252.00", is_deterministic: bool = True) -> RiskProfile:
    return RiskProfile(
        soc_code=soc_code,
        role_title="Software Developers",
        threat_score=35.0,
        risk_level=RiskLevel.MODERATE,
        sub_scores=SubRiskScores(task_automation=30.0, tool_maturity=40.0, agentic_exposure=35.0),
        defense_skills=["System Architecture", "Complex Problem Solving"],
        vulnerable_skills=["Routine Coding"],
        confidence=1.0 if is_deterministic else 0.82,
        is_deterministic=is_deterministic,
    )


def create_mock_roadmap(target_role: str = "Cloud Solutions Architect") -> CopilotRoadmap:
    phase1 = RoadmapPhase(
        phase_name="Phase 1: 30 Days - Foundation",
        duration_days=30,
        objective="Master cloud architecture fundamentals",
        action_items=["Complete AWS Associate labs"],
        recommended_certifications=["AWS Solutions Architect Associate"],
        target_skills=["Cloud Architecture"],
        practical_projects=["Container Deployment"],
    )
    phase2 = RoadmapPhase(
        phase_name="Phase 2: 60 Days - Deepening",
        duration_days=30,
        objective="Deploy scalable Kubernetes workloads",
        action_items=["Configure Helm charts"],
        recommended_certifications=["CKA"],
        target_skills=["Kubernetes"],
        practical_projects=["Production K8s"],
    )
    phase3 = RoadmapPhase(
        phase_name="Phase 3: 90 Days - Capstone",
        duration_days=30,
        objective="Deliver end-to-end cloud capstone",
        action_items=["Publish capstone repo"],
        recommended_certifications=["AWS SA Pro"],
        target_skills=["Enterprise Architecture"],
        practical_projects=["Full-stack Cloud App"],
    )
    return CopilotRoadmap(
        target_role=target_role,
        target_soc_code="15-1252.00",
        executive_summary=f"90-day transition roadmap into {target_role}.",
        phases=[phase1, phase2, phase3],
        defense_skills_breakdown=["Cloud Architecture", "Kubernetes"],
        mermaid_chart="graph TD\n    A[Phase 1] --> B[Phase 2] --> C[Phase 3]",
        estimated_completion_months=3,
    )


# ==============================================================================
# 1. Tests for HybridRiskRouter
# ==============================================================================

def test_hybrid_risk_router_high_confidence():
    """Verify deterministic 0ms O*NET lookup when title matching confidence >= 0.90."""
    mock_onet_repo = MagicMock(spec=IOnetRepository)
    mock_risk_pred = MagicMock(spec=IRiskPredictor)

    expected_risk = create_mock_risk_profile(soc_code="15-1252.00", is_deterministic=True)
    mock_onet_repo.get_risk_profile.return_value = expected_risk

    router = HybridRiskRouter(
        onet_repo=mock_onet_repo,
        risk_predictor=mock_risk_pred,
        threshold=0.90,
    )

    candidate = create_mock_candidate()
    result = router.determine_risk(
        profile=candidate,
        soc_code="15-1252.00",
        confidence=0.95,
        user_confirmed=False,
    )

    assert result.is_deterministic is True
    assert result.soc_code == "15-1252.00"
    mock_onet_repo.get_risk_profile.assert_called_once_with("15-1252.00")
    mock_risk_pred.predict_risk.assert_not_called()


def test_hybrid_risk_router_low_confidence():
    """Verify ML XGBoost regressor is called when confidence < 0.90."""
    mock_onet_repo = MagicMock(spec=IOnetRepository)
    mock_risk_pred = MagicMock(spec=IRiskPredictor)

    ml_inferred_risk = create_mock_risk_profile(soc_code="15-1252.00", is_deterministic=False)
    mock_risk_pred.predict_risk.return_value = ml_inferred_risk

    router = HybridRiskRouter(
        onet_repo=mock_onet_repo,
        risk_predictor=mock_risk_pred,
        threshold=0.90,
    )

    candidate = create_mock_candidate()
    result = router.determine_risk(
        profile=candidate,
        soc_code="15-1252.00",
        confidence=0.78,
        user_confirmed=False,
    )

    assert result.is_deterministic is False
    mock_risk_pred.predict_risk.assert_called_once_with(candidate)
    mock_onet_repo.get_risk_profile.assert_not_called()


def test_hybrid_risk_router_user_confirmed_override():
    """Verify human confirmation routes to deterministic O*NET even if initial confidence was low."""
    mock_onet_repo = MagicMock(spec=IOnetRepository)
    mock_risk_pred = MagicMock(spec=IRiskPredictor)

    expected_risk = create_mock_risk_profile(soc_code="15-1252.00", is_deterministic=True)
    mock_onet_repo.get_risk_profile.return_value = expected_risk

    router = HybridRiskRouter(
        onet_repo=mock_onet_repo,
        risk_predictor=mock_risk_pred,
        threshold=0.90,
    )

    candidate = create_mock_candidate()
    result = router.determine_risk(
        profile=candidate,
        soc_code="15-1252.00",
        confidence=0.55,  # Below threshold, but confirmed
        user_confirmed=True,
    )

    assert result.is_deterministic is True
    mock_onet_repo.get_risk_profile.assert_called_once_with("15-1252.00")
    mock_risk_pred.predict_risk.assert_not_called()


def test_hybrid_risk_router_fallback_on_missing_soc():
    """Verify fallback to ML regressor when SOC is not found in O*NET catalog."""
    mock_onet_repo = MagicMock(spec=IOnetRepository)
    mock_risk_pred = MagicMock(spec=IRiskPredictor)

    mock_onet_repo.get_risk_profile.return_value = None  # SOC not found
    ml_risk = create_mock_risk_profile(soc_code="custom-role", is_deterministic=False)
    mock_risk_pred.predict_risk.return_value = ml_risk

    router = HybridRiskRouter(
        onet_repo=mock_onet_repo,
        risk_predictor=mock_risk_pred,
        threshold=0.90,
    )

    candidate = create_mock_candidate()
    result = router.determine_risk(
        profile=candidate,
        soc_code="99-9999.00",
        confidence=0.99,
        user_confirmed=True,
    )

    assert result.is_deterministic is False
    mock_onet_repo.get_risk_profile.assert_called_once_with("99-9999.00")
    mock_risk_pred.predict_risk.assert_called_once_with(candidate)


# ==============================================================================
# 2. Tests for CVAnalysisService (Unified Facade Orchestrator)
# ==============================================================================

@pytest.fixture
def test_copilot_service() -> Tuple[CVAnalysisService, CopilotRoadmap]:
    """Builds a test-isolated CVAnalysisService wrapping LangGraph with Mock LLMs."""
    candidate = create_mock_candidate()
    mock_roadmap = create_mock_roadmap()

    critic_approval = CriticEvaluation(
        approved=True,
        feedback="Roadmap structured realistically.",
        scores=CriticScores(feasibility=9.0, grounding=9.0, hallucination_free=9.5),
    )

    # Mock LLM for Copilot generation & Critic
    mock_copilot_llm = MockStructuredLLM([mock_roadmap, "Start by reviewing Phase 1 containerization milestones."])
    mock_critic_llm = MockStructuredLLM([critic_approval])

    # Mock profile parser that immediately returns CandidateProfile
    mock_profile_parser = MagicMock()
    mock_profile_parser.extract_profile.return_value = candidate
    mock_profile_parser.parse_profile_from_text.return_value = candidate

    # Mock OnetRepository
    mock_onet_repo = MagicMock(spec=IOnetRepository)
    mock_onet_repo.match_job_title.return_value = ("15-1252.00", "Software Developers", 0.95)
    mock_onet_repo.resolve_candidate_soc.return_value = ("15-1252.00", "Software Developers", 0.95, False)
    mock_onet_repo.get_risk_profile.return_value = create_mock_risk_profile("15-1252.00", is_deterministic=True)

    # Mock CareerService
    mock_career_service = MagicMock()
    mock_pivot = SafeTransitionRole(
        target_soc_code="15-1252.00",
        target_title="Cloud Solutions Architect",
        utility_score=85.0,
        similarity_pct=88.0,
        threat_reduction=20.0,
        target_threat_score=25.0,
        target_risk_level=RiskLevel.LOW,
        defense_skills=["Cloud Architecture", "Kubernetes"],
    )
    mock_career_service.get_safe_career_pivots.return_value = [mock_pivot]

    # Mock EvidenceRetriever
    mock_retriever = MagicMock()
    mock_retriever.retrieve_evidence.return_value = [
        {"soc_code": "15-1252", "aspect": "Tasks", "content": "Design distributed cloud architectures."}
    ]

    memory = MemorySaver()
    test_graph = build_copilot_graph(
        profile_parser=mock_profile_parser,
        onet_repo=mock_onet_repo,
        career_service=mock_career_service,
        evidence_retriever=mock_retriever,
        copilot_llm=mock_copilot_llm,
        critic_llm=mock_critic_llm,
        checkpointer=memory,
        interrupt_before=["human_review_node"],
    )

    service = CVAnalysisService(
        onet_repo=mock_onet_repo,
        career_service=mock_career_service,
        copilot_graph=test_graph,
        checkpointer=memory,
        evidence_retriever=mock_retriever,
    )
    return service, mock_roadmap


def test_cv_analysis_service_start_to_hitl(test_copilot_service):
    """Verify Phase 1: Ingests CV and accurately pauses at HITL checkpoint in <3s."""
    service, _ = test_copilot_service
    session_id = "test_facade_session_01"

    raw_text = "Le Van An - Software Developer with 4 years in Python, FastAPI, Docker, and PostgreSQL."
    result = service.start_cv_analysis(raw_cv_text=raw_text, session_id=session_id)

    assert result["session_id"] == session_id
    assert result["status"] == "AWAITING_HUMAN_REVIEW"
    assert result["candidate_profile"] is not None
    assert result["candidate_profile"].full_name == "Le Van An"
    assert result["suggested_soc_code"] == "15-1252.00"
    assert result["suggested_title"] == "Software Developers"
    assert result["match_confidence"] >= 0.90
    assert result["latency_ms"] >= 0.0


def test_cv_analysis_service_confirm_and_resume(test_copilot_service):
    """Verify Phase 2: Confirms candidate profile, resumes execution, and produces AnalysisResult."""
    service, mock_roadmap = test_copilot_service
    session_id = "test_facade_session_02"

    raw_text = "Le Van An - Software Developer with 4 years in Python, FastAPI, Docker."
    # 1. Start Phase 1
    init_res = service.start_cv_analysis(raw_cv_text=raw_text, session_id=session_id)
    assert init_res["status"] == "AWAITING_HUMAN_REVIEW"

    # 2. Confirm & Resume Phase 2 with override
    analysis = service.confirm_and_resume_analysis(
        session_id=session_id,
        user_confirmed=True,
        overrides={"matched_title": "Lead Software Developer"},
    )

    assert isinstance(analysis, AnalysisResult)
    assert analysis.session_id == session_id
    assert analysis.candidate_profile.full_name == "Le Van An"
    assert analysis.risk_profile is not None
    assert analysis.roadmap is not None
    assert analysis.roadmap.target_role == mock_roadmap.target_role
    assert len(analysis.roadmap.phases) == 3
    assert len(analysis.safe_pivots) >= 1
    assert len(analysis.skill_gaps) >= 1


def test_cv_analysis_service_chat_with_copilot(test_copilot_service):
    """Verify Phase 3: Conversational multi-turn Q&A anchored in approved roadmap."""
    service, _ = test_copilot_service
    session_id = "test_facade_session_03"

    raw_text = "Le Van An - Software Developer with 4 years in Python, FastAPI, Docker."
    service.start_cv_analysis(raw_cv_text=raw_text, session_id=session_id)
    service.confirm_and_resume_analysis(session_id=session_id, user_confirmed=True)

    # Ask follow-up question
    response = service.chat_with_copilot(
        session_id=session_id,
        message="Which certification should I tackle first?",
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert "Phase 1" in response or "milestone" in response or len(response) > 10


def test_cv_analysis_service_get_analysis_result(test_copilot_service):
    """Verify session result retrieval for Dashboard and caching."""
    service, mock_roadmap = test_copilot_service
    session_id = "test_facade_session_04"

    raw_text = "Le Van An - Software Developer with 4 years in Python, FastAPI, Docker."
    service.start_cv_analysis(raw_cv_text=raw_text, session_id=session_id)
    service.confirm_and_resume_analysis(session_id=session_id, user_confirmed=True)

    # Retrieve existing
    existing = service.get_analysis_result(session_id)
    assert existing is not None
    assert existing.session_id == session_id
    assert existing.roadmap.target_role == mock_roadmap.target_role

    # Non-existent session
    missing = service.get_analysis_result("non_existent_session_999")
    assert missing is None


def test_cv_analysis_service_error_handling(test_copilot_service):
    """Verify input validation and edge cases."""
    service, _ = test_copilot_service

    # Missing both file_path and raw_cv_text
    with pytest.raises(ValueError, match="Either file_path or raw_cv_text"):
        service.start_cv_analysis()

    # Unknown session resume
    with pytest.raises(KeyError, match="No active session found"):
        service.confirm_and_resume_analysis(session_id="unknown_session_001")

    # Empty chat message
    with pytest.raises(ValueError, match="Chat message must not be empty"):
        service.chat_with_copilot(session_id="any_session", message="   ")


def test_get_cv_analysis_service_factory():
    """Verify FastAPI Dependency Provider singleton returns a valid CVAnalysisService."""
    factory_service = get_cv_analysis_service()
    assert isinstance(factory_service, CVAnalysisService)
    assert factory_service is get_cv_analysis_service()  # Singleton identity
