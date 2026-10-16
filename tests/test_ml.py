"""
Unit & Integration Tests for Machine Learning Core (Task 04).
Tests AgglomerativeClusteringEngine and XGBoostRiskPredictor protocols,
persistence, SHAP explainability, and transition utility logic.
"""

from pathlib import Path
import tempfile
import pytest
import numpy as np

from source.domain.interfaces.ml_models import IClusteringEngine, IRiskPredictor
from source.domain.entities.candidate import CandidateProfile, WorkExperience
from source.domain.entities.risk import RiskProfile, RiskLevel
from source.domain.entities.transition import SafeTransitionRole, SkillGap
from source.domain.exceptions import RoleNotFoundError
from source.model.ML_model.clustering_engine import AgglomerativeClusteringEngine
from source.model.ML_model.risk_predictor import XGBoostRiskPredictor


@pytest.fixture
def sample_candidate_profile() -> CandidateProfile:
    """Fixture providing a realistic AI/Software Engineer profile."""
    return CandidateProfile(
        full_name="Nguyen Hai Nam",
        current_title="AI Engineer",
        years_of_experience=3.0,
        hard_skills=["Python", "Machine Learning", "PyTorch", "RAG", "Vector Search", "FastAPI"],
        soft_skills=["Critical Thinking", "Problem Solving", "Communication"],
        tools_and_technologies=["Docker", "PostgreSQL", "MongoDB Atlas", "Git", "LangChain"],
        experiences=[
            WorkExperience(
                company="AI Innovation Lab",
                role_title="AI Software Engineer",
                duration_months=36,
                description="Built ReAct Agent and RAG pipeline using vector embeddings. Optimized cosine similarity search and integrated Fast LLMs.",
                key_achievements=["Deployed agent into production with sub-second latency."],
            )
        ],
        education=["B.S. in Computer Science - HUST"],
    )


# ==============================================================================
# 1. Tests for AgglomerativeClusteringEngine
# ==============================================================================

def test_clustering_engine_protocol_conformance():
    """Verify runtime protocol check for IClusteringEngine."""
    engine = AgglomerativeClusteringEngine()
    assert isinstance(engine, IClusteringEngine)


def test_clustering_engine_fit_and_metrics():
    """Verify model fitting, metric calculation, and balanced cluster sizes."""
    engine = AgglomerativeClusteringEngine(n_clusters=23)
    engine.fit()

    assert engine.is_fitted
    assert len(engine.soc_codes) == 867
    assert engine.normalized_matrix.shape == (867, 76)
    assert engine.silhouette_ > 0.15
    assert engine.nmi_ > 0.35  # Spherical Ward achieves ~0.466

    cluster_sizes = [len(socs) for socs in engine.cluster_to_socs.values()]
    assert max(cluster_sizes) < 150  # No giant cluster collapse (contrast with 743 in average linkage)
    assert min(cluster_sizes) >= 2


def test_clustering_engine_safe_pivots_valid_soc():
    """Test safe career pivot discovery for Software Developers (15-1252)."""
    engine = AgglomerativeClusteringEngine()
    engine.fit()

    soc_code = "15-1252"  # Software Developers (normalized 6-digit)
    current_threat = float(engine.metadata.loc[soc_code]["threat_score"])

    # Test both '15-1252' and '15-1252.00'
    pivots = engine.get_safe_pivots(soc_code=soc_code, current_threat=current_threat, top_k=5)

    assert len(pivots) > 0
    assert len(pivots) <= 5

    top_pivot = pivots[0]
    assert isinstance(top_pivot, SafeTransitionRole)
    assert top_pivot.target_soc_code != soc_code
    assert top_pivot.target_threat_score <= current_threat
    assert 0.0 <= top_pivot.similarity_pct <= 100.0
    assert 0.0 <= top_pivot.utility_score <= 100.0

    # Test normalization when passing '15-1252.00'
    pivots_dotted = engine.get_safe_pivots(soc_code="15-1252.00", current_threat=current_threat, top_k=3)
    assert len(pivots_dotted) > 0


def test_clustering_engine_safe_pivots_invalid_soc():
    """Test that RoleNotFoundError is raised when SOC code is missing."""
    engine = AgglomerativeClusteringEngine()
    engine.fit()

    with pytest.raises(RoleNotFoundError):
        engine.get_safe_pivots("99-9999.99", current_threat=50.0)


def test_clustering_engine_serialization(tmp_path: Path):
    """Test saving and loading clustering engine artifact."""
    engine = AgglomerativeClusteringEngine(n_clusters=23)
    engine.fit()

    save_file = tmp_path / "test_cluster.joblib"
    engine.save_model(save_file)
    assert save_file.exists()

    new_engine = AgglomerativeClusteringEngine()
    new_engine.load_model(save_file)

    assert new_engine.is_fitted
    assert len(new_engine.soc_codes) == 867
    assert new_engine.n_clusters == 23


# ==============================================================================
# 2. Tests for XGBoostRiskPredictor
# ==============================================================================

def test_risk_predictor_protocol_conformance():
    """Verify runtime protocol check for IRiskPredictor."""
    predictor = XGBoostRiskPredictor()
    assert isinstance(predictor, IRiskPredictor)


def test_risk_predictor_fit_and_predict(sample_candidate_profile: CandidateProfile):
    """Verify training, prediction, deterministic mapping, and SHAP explainability."""
    predictor = XGBoostRiskPredictor()
    predictor.fit()

    assert predictor.is_fitted
    assert len(predictor.feature_names) == 76
    assert predictor.explainer is not None

    risk = predictor.predict_risk(sample_candidate_profile)

    assert isinstance(risk, RiskProfile)
    assert 0.0 <= risk.threat_score <= 100.0
    assert risk.risk_level in list(RiskLevel)
    assert not risk.is_deterministic
    assert len(risk.defense_skills) == 3
    assert len(risk.insight) > 20
    assert 0.0 <= risk.sub_scores.task_automation <= 100.0
    assert 0.0 <= risk.sub_scores.tool_maturity <= 100.0
    assert 0.0 <= risk.sub_scores.agentic_exposure <= 100.0


def test_risk_predictor_vectorize_profile(sample_candidate_profile: CandidateProfile):
    """Test mapping candidate profile into 76D capability vector."""
    predictor = XGBoostRiskPredictor()
    predictor.fit()

    vec = predictor.vectorize_profile(sample_candidate_profile)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (76,)
    assert (vec >= 0.0).all() and (vec <= 1.0).all()
    # At least some technical features should be activated above baseline
    assert vec.max() > 0.40


def test_risk_predictor_deterministic_boundary_mapping():
    """Verify threshold mapping for RiskLevel."""
    assert XGBoostRiskPredictor.map_score_to_risk_level(15.0) == RiskLevel.LOW
    assert XGBoostRiskPredictor.map_score_to_risk_level(45.0) == RiskLevel.MODERATE
    assert XGBoostRiskPredictor.map_score_to_risk_level(65.0) == RiskLevel.HIGH
    assert XGBoostRiskPredictor.map_score_to_risk_level(85.0) == RiskLevel.VERY_HIGH


def test_risk_predictor_serialization(tmp_path: Path, sample_candidate_profile: CandidateProfile):
    """Test saving and loading XGBoost risk predictor artifact."""
    predictor = XGBoostRiskPredictor()
    predictor.fit()

    save_file = tmp_path / "test_xgb.joblib"
    predictor.save_model(save_file)
    assert save_file.exists()

    new_predictor = XGBoostRiskPredictor()
    new_predictor.load_model(save_file)

    assert new_predictor.is_fitted
    assert len(new_predictor.feature_names) == 76

    risk = new_predictor.predict_risk(sample_candidate_profile)
    assert isinstance(risk, RiskProfile)
    assert 0.0 <= risk.threat_score <= 100.0
