"""
Unit & Integration Tests for UserCareerService (source/services/user/career_service.py).
Tests lazy loading from models/, safe pivot orchestration,
risk prediction, and candidate transition analysis.
"""

from pathlib import Path
import pytest

from source.domain.entities.candidate import CandidateProfile, WorkExperience
from source.domain.entities.risk import RiskProfile, RiskLevel, SubRiskScores
from source.domain.entities.transition import SafeTransitionRole, SkillGap
from source.services.user.career_service import UserCareerService
from source.domain.interfaces.ml_models import IClusteringEngine, IRiskPredictor


@pytest.fixture
def candidate_hai_nam() -> CandidateProfile:
    """Fixture mimicking user's actual CV from CV Nguyen Hai Nam.pdf."""
    return CandidateProfile(
        full_name="Nguyen Hai Nam",
        current_title="AI / Software Engineering Candidate",
        years_of_experience=2.5,
        hard_skills=[
            "Machine Learning", "ReAct Agent", "RAG Pipeline",
            "Vector Search", "Cosine Similarity", "Python", "SQL Optimization"
        ],
        soft_skills=["Problem Solving", "Continuous Learning", "Teamwork"],
        tools_and_technologies=[
            "Docker", "PostgreSQL", "MongoDB Atlas", "LangChain", "PyMuPDF", "Git"
        ],
        experiences=[
            WorkExperience(
                company="AI Development Project",
                role_title="AI Engineer / Researcher",
                duration_months=18,
                description="Developed RAG architecture using MongoDB Atlas vector search. Implemented ReAct Agent workflow with Fast LLM APIs.",
                key_achievements=["Deployed agent into production."],
            )
        ],
        education=["Computer Science - Hanoi University of Science and Technology (HUST)"],
    )


class MockClusteringEngine(IClusteringEngine):
    """Mock clustering engine returning controlled safe pivots."""

    def __init__(self):
        self.called_with_soc = None

    def get_safe_pivots(self, soc_code: str, current_threat: float, top_k: int = 5) -> list[SafeTransitionRole]:
        self.called_with_soc = soc_code
        return [
            SafeTransitionRole(
                target_soc_code="15-1253",
                target_title="Software Quality Assurance Analyst",
                utility_score=85.0,
                similarity_pct=88.0,
                threat_reduction=10.0,
                salary_advantage=0.05,
                target_threat_score=18.0,
                matched_skills=["Programming", "Critical Thinking"],
                missing_critical_skills=["Quality Control Analysis"],
            )
        ]


class MockRiskPredictor(IRiskPredictor):
    """Mock risk predictor returning controlled risk profile."""

    def __init__(self):
        self.called = False

    def predict_risk(self, profile: CandidateProfile) -> RiskProfile:
        self.called = True
        return RiskProfile(
            soc_code=None,
            role_title="Senior Python Developer",
            threat_score=22.5,
            risk_level=RiskLevel.LOW,
            sub_scores=SubRiskScores(
                task_automation=20.0,
                tool_maturity=24.0,
                agentic_exposure=19.0,
            ),
            defense_skills=["Programming", "Critical Thinking", "Complex Problem Solving"],
            insight="Mock analysis shows low automation threat.",
            is_deterministic=False,
        )


class MockOnetRepo:
    def match_job_title(self, raw_title: str):
        return ("15-1252.00", "Software Developers", 0.95)

    def get_risk_profile(self, soc_code: str):
        return RiskProfile(
            soc_code=soc_code,
            role_title="Software Developers",
            threat_score=28.0,
            risk_level=RiskLevel.LOW,
            is_deterministic=True,
        )

    def get_feature_vector(self, soc_code: str):
        return None

    def get_metadata(self, soc_code: str):
        return {"threat_score": 28.0, "soc_title": "Software Developers"}


# ==============================================================================
# Tests for UserCareerService
# ==============================================================================

def test_user_career_service_dependency_injection():
    """Verify service behavior with injected mock dependencies (zero disk access)."""
    mock_cluster = MockClusteringEngine()
    mock_risk = MockRiskPredictor()
    mock_repo = MockOnetRepo()

    service = UserCareerService(
        onet_repo=mock_repo,
        clustering_engine=mock_cluster,
        risk_predictor=mock_risk,
    )

    # 1. Test Safe Pivots without passing current_threat (service delegates to onet_repo)
    pivots = service.get_safe_career_pivots("15-1252.00")
    assert len(pivots) == 1
    assert mock_cluster.called_with_soc == "15-1252.00"
    assert pivots[0].target_soc_code == "15-1253"

    # 2. Test Risk Prediction
    profile = CandidateProfile(full_name="Test User", current_title="Developer")
    risk = service.predict_hybrid_role_risk(profile)
    assert mock_risk.called
    assert risk.threat_score == 22.5
    assert risk.risk_level == RiskLevel.LOW


def test_user_career_service_lazy_load_from_disk(candidate_hai_nam: CandidateProfile):
    """
    Integration test: loads actual trained artifacts from models/
    and executes full candidate transition analysis on user's real-world profile.
    """
    service = UserCareerService()

    # Step 1: Predict Risk Profile on real profile
    risk = service.predict_hybrid_role_risk(candidate_hai_nam)
    assert isinstance(risk, RiskProfile)
    assert 0.0 <= risk.threat_score <= 100.0
    assert len(risk.defense_skills) == 3
    assert not risk.is_deterministic

    # Step 2: Safe Pivots for Software Developers (15-1252.00)
    pivots = service.get_safe_career_pivots(soc_code="15-1252.00", top_k=3)
    assert len(pivots) > 0
    assert pivots[0].target_threat_score <= 28.0  # Must be safer than current software developers threat

    # Step 3: Full End-to-End Transition Analysis
    result = service.analyze_candidate_transition(
        profile=candidate_hai_nam,
        current_soc_code="15-1252.00",
        top_k=3,
    )

    assert result["candidate_name"] == "Nguyen Hai Nam"
    assert result["matched_soc_code"] == "15-1252.00"
    assert isinstance(result["risk_profile"], RiskProfile)
    assert len(result["safe_transition_roles"]) > 0
    assert len(result["top_defense_skills"]) == 3
    assert result["analysis_latency_ms"] > 0
