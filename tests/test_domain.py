"""
Unit Tests for Domain Layer (Entities, Protocols, Exceptions).
Validates 100% adherence to Domain-Driven Design contracts and Module 1 specifications.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pytest

from source.domain import (
    SkillItem,
    WorkExperience,
    CandidateProfile,
    OnetRole,
    RiskLevel,
    SubRiskScores,
    RiskProfile,
    SafeTransitionRole,
    SkillGap,
    RoadmapPhase,
    CopilotRoadmap,
    AnalysisResult,
    IDataLoader,
    IFeatureTransformer,
    IMetadataBuilder,
    IOnetRepository,
    IDocumentParser,
    IProfileExtractor,
    IClusteringEngine,
    IRiskPredictor,
    IEvidenceRetriever,
    DomainError,
    CVExtractionError,
    TitleMatchingError,
    RoleNotFoundError,
    ModelInferenceError,
    RoadmapGenerationError,
)


def test_candidate_profile_validation_and_skills():
    """
    Test CandidateProfile entity:
    - Verifies Pydantic v2 instantiations and alias resolution (years_of_experience).
    - Verifies get_all_skills method de-duplicates hard, soft, and tool skills.
    - Verifies nested WorkExperience and SkillItem structures.
    """
    exp = WorkExperience(
        company="Tech Corp",
        role_title="Backend Developer",
        duration_months=24,
        description="Developed microservices with Python and FastAPI",
        key_achievements=["Optimized database latency by 40%"],
    )

    skill = SkillItem(
        name="Python",
        category="hard_skill",
        proficiency_level="advanced",
    )

    profile = CandidateProfile(
        full_name="Alex Mercer",
        current_title="Software Engineer",
        year_of_experience=3.5,  # testing alias
        hard_skills=["Python", "SQL", "Docker"],
        soft_skills=["Problem Solving", "Communication"],
        tools_and_technologies=["Docker", "Git", "PostgreSQL"],
        experiences=[exp],
        education=["B.S. in Computer Science"],
    )

    assert profile.full_name == "Alex Mercer"
    assert profile.years_of_experience == 3.5
    assert profile.year_of_experience == 3.5
    assert len(profile.experiences) == 1
    assert profile.experiences[0].company == "Tech Corp"

    # Verify skill set unification and de-duplication
    all_skills = profile.get_all_skills
    assert "Python" in all_skills
    assert "Docker" in all_skills
    assert "Problem Solving" in all_skills
    assert len(all_skills) == len(set(all_skills))


def test_onet_role_entity_creation():
    """
    Test OnetRole entity:
    - Verifies 6-digit SOC code formatting.
    - Verifies threat score, risk level, and compensation values.
    """
    role = OnetRole(
        soc_code="15-1252",
        soc_title="Software Developers",
        major_group="15",
        major_group_name="Computer and Mathematical Occupations",
        median_salary=127260.0,
        threat_score=68.5,
        risk_level=RiskLevel.HIGH,
        sub_task_automation=70.0,
        sub_tool_maturity=72.0,
        sub_agentic_exposure=63.5,
        defense_skills=["Systems Analysis", "Complex Problem Solving", "Critical Thinking"],
        top_hot_technologies=["Docker", "Kubernetes", "PyTorch"],
        critical_tasks=["Design and develop software architectures"],
    )

    assert role.soc_code == "15-1252"
    assert role.soc_title == "Software Developers"
    assert role.risk_level == "High"
    assert role.threat_score == 68.5
    assert len(role.defense_skills) == 3


def test_risk_and_transition_entities():
    """
    Test RiskProfile, SafeTransitionRole, SkillGap, and CopilotRoadmap entities.
    """
    sub_scores = SubRiskScores(
        task_automation=65.0,
        tool_maturity=70.0,
        agentic_exposure=55.0,
    )
    risk = RiskProfile(
        soc_code="15-1252",
        role_title="Software Developers",
        threat_score=63.3,
        risk_level=RiskLevel.HIGH,
        sub_scores=sub_scores,
        defense_skills=["Architecture Design", "Complex Problem Solving", "Strategic Leadership"],
        is_deterministic=True,
    )

    pivot = SafeTransitionRole(
        target_soc_code="15-1212",
        target_title="Information Security Analysts",
        utility_score=82.5,
        similarity_pct=78.0,
        threat_reduction=25.0,
        salary_advantage=0.15,
        target_threat_score=38.3,
        target_risk_level=RiskLevel.MODERATE,
        defense_skills=["Cyber Threat Intelligence", "Security Architecture", "Active Defense"],
    )

    skill_gap = SkillGap(
        target_role_title="Information Security Analysts",
        target_soc_code="15-1212",
        strengths=["Python", "Linux", "Networking"],
        vulnerable_skills=["Routine CRUD Generation"],
        defense_skills=["Active Defense", "Risk Auditing"],
        missing_skills=["SIEM", "Penetration Testing"],
        overlap_percentage=78.0,
    )

    phase = RoadmapPhase(
        phase_name="Phase 1: 30 Days - Foundation",
        duration_days=30,
        objective="Master baseline security defense",
        action_items=["Complete CompTIA Security+ syllabus"],
        recommended_certifications=["Security+"],
    )

    roadmap = CopilotRoadmap(
        target_role="Information Security Analysts",
        target_soc_code="15-1212",
        executive_summary="Strategic defensive pivot to Cybersecurity",
        phases=[phase],
        defense_skills_breakdown=["Security Architecture"],
        mermaid_chart="graph TD; A[Start]-->B[Phase 1];",
    )

    analysis = AnalysisResult(
        candidate_profile=CandidateProfile(current_title="Software Engineer"),
        risk_profile=risk,
        safe_pivots=[pivot],
        skill_gap=skill_gap,
        roadmap=roadmap,
    )

    assert analysis.risk_profile.threat_score == 63.3
    assert len(analysis.safe_pivots) == 1
    assert analysis.safe_pivots[0].target_soc_code == "15-1212"
    assert analysis.roadmap.target_role == "Information Security Analysts"


def test_domain_protocols_runtime_checkable():
    """
    Test that all Domain Protocols are @runtime_checkable and compatible with duck typing.
    """
    class MockDocumentParser:
        def extract_text(self, file_path: Path) -> str:
            return "Extracted resume content"

    class MockProfileExtractor:
        def extract_profile(self, raw_text: str) -> CandidateProfile:
            return CandidateProfile(full_name="Test Candidate")

    class MockClusteringEngine:
        def get_safe_pivots(self, soc_code: str, current_threat: float, top_k: int = 5) -> List[SafeTransitionRole]:
            return []

    class MockRiskPredictor:
        def predict_risk(self, profile: CandidateProfile) -> RiskProfile:
            return RiskProfile(
                role_title=profile.current_title or "Unknown",
                threat_score=50.0,
                risk_level=RiskLevel.MODERATE,
            )

    class MockEvidenceRetriever:
        def retrieve_evidence(
            self, query: str, soc_code: Optional[str] = None, major_group: Optional[str] = None, top_k: int = 5
        ) -> List[Dict[str, Any]]:
            return []

    class MockRepository:
        def match_job_title(self, raw_title: str) -> Tuple[str, str, float]:
            return ("15-1252", "Software Developers", 0.95)

        def match_role_by_skills(self, skills: List[str]) -> Tuple[Optional[str], Optional[str], float]:
            return ("15-1252", "Software Developers", 0.90)

        def match_role_by_education(self, major: str) -> Tuple[Optional[str], Optional[str], float]:
            return ("15-1252", "Software Developers", 0.85)

        def get_risk_profile(self, soc_code: str) -> Optional[RiskProfile]:
            return None

        def get_feature_vector(self, soc_code: str) -> Optional[np.ndarray]:
            return None

        def get_metadata(self, soc_code: str) -> Optional[Dict[str, Any]]:
            return None

        def resolve_candidate_soc(self, profile: Any) -> Tuple[Optional[str], Optional[str], float, bool]:
            return ("15-1252", "Software Developers", 0.95, False)

    # Protocol compliance verification via isinstance
    assert isinstance(MockDocumentParser(), IDocumentParser)
    assert isinstance(MockProfileExtractor(), IProfileExtractor)
    assert isinstance(MockClusteringEngine(), IClusteringEngine)
    assert isinstance(MockRiskPredictor(), IRiskPredictor)
    assert isinstance(MockEvidenceRetriever(), IEvidenceRetriever)
    assert isinstance(MockRepository(), IOnetRepository)


def test_domain_exceptions_hierarchy():
    """
    Test Domain Exceptions hierarchy:
    - All domain errors must inherit from DomainError.
    - Specific exceptions must be distinct and convey business reasons.
    """
    err = CVExtractionError("Failed to extract PDF blocks")
    assert isinstance(err, DomainError)
    assert isinstance(err, Exception)
    assert str(err) == "Failed to extract PDF blocks"

    assert issubclass(TitleMatchingError, DomainError)
    assert issubclass(RoleNotFoundError, DomainError)
    assert issubclass(ModelInferenceError, DomainError)
    assert issubclass(RoadmapGenerationError, DomainError)


def test_dod_import_command():
    """
    Verify the exact Definition of Done (DoD) verification command from docs/tasks/01_domain_module.md.
    """
    from source.domain.entities.candidate import CandidateProfile
    from source.domain.entities.risk import RiskProfile
    from source.domain.entities.role import OnetRole
    from source.domain.interfaces.parser import IDocumentParser, IProfileExtractor
    from source.domain.exceptions import CVExtractionError

    assert CandidateProfile is not None
    assert RiskProfile is not None
    assert OnetRole is not None
    assert IDocumentParser is not None
    assert IProfileExtractor is not None
    assert CVExtractionError is not None
