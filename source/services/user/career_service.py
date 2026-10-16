"""
User Career Transition & Risk Analysis Service (Use Case Layer).
Pure Orchestrator adhering strictly to Clean Architecture:
- Coordinates IOnetRepository (Database), IClusteringEngine (ML), and IRiskPredictor (ML).
- 0% direct filesystem / parquet reads: All database access is delegated to IOnetRepository.
"""

import time
from typing import List, Dict, Any, Optional

from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.risk import RiskProfile
from source.domain.entities.transition import SafeTransitionRole
from source.domain.interfaces.onet_repository import IOnetRepository
from source.domain.interfaces.ml_models import IClusteringEngine, IRiskPredictor
from source.infrastructure.database.repo.onet_repository import OnetRepository
from source.model.ML_model.clustering_engine import AgglomerativeClusteringEngine
from source.model.ML_model.risk_predictor import XGBoostRiskPredictor
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class UserCareerService:
    """
    Application Service orchestrating ML Core models and Database Repository
    for user-facing career transition advising workflows.
    Decoupled via IOnetRepository, IClusteringEngine, and IRiskPredictor protocols.
    """

    def __init__(
        self,
        onet_repo: Optional[IOnetRepository] = None,
        clustering_engine: Optional[IClusteringEngine] = None,
        risk_predictor: Optional[IRiskPredictor] = None,
    ):
        self._onet_repo = onet_repo
        self._clustering_engine = clustering_engine
        self._risk_predictor = risk_predictor

        logger.debug("UserCareerService orchestrator initialized.")

    @property
    def onet_repo(self) -> IOnetRepository:
        """Lazy-loads or initializes IOnetRepository implementation from infrastructure."""
        if self._onet_repo is None:
            self._onet_repo = OnetRepository()
        return self._onet_repo

    @property
    def clustering_engine(self) -> IClusteringEngine:
        """Lazy-loads or initializes AgglomerativeClusteringEngine from disk."""
        if self._clustering_engine is None:
            engine = AgglomerativeClusteringEngine(
                n_clusters=settings.CLUSTERING_N_CLUSTERS,
            )
            model_file = settings.MODEL_DIR / "clustering_model.joblib"
            if model_file.exists():
                try:
                    meta_df = self.onet_repo.get_metadata_dataframe()
                    engine.load_model(model_file, metadata=meta_df if not meta_df.empty else None)
                except Exception as e:
                    logger.warning("Failed to load clustering model from '%s' (%s). Refitting...", model_file, e)
                    engine.fit()
                    engine.save_model(model_file)
            else:
                logger.info("Clustering artifact not found at '%s'. Fitting fresh model...", model_file)
                engine.fit()
                engine.save_model(model_file)

            self._clustering_engine = engine
        return self._clustering_engine

    @property
    def risk_predictor(self) -> IRiskPredictor:
        """Lazy-loads or initializes XGBoostRiskPredictor from disk."""
        if self._risk_predictor is None:
            predictor = XGBoostRiskPredictor(
                model_path=settings.MODEL_DIR / "xgb_risk_model.joblib",
            )
            model_file = settings.MODEL_DIR / "xgb_risk_model.joblib"
            if model_file.exists():
                try:
                    predictor.load_model(model_file)
                except Exception as e:
                    logger.warning("Failed to load XGBoost model from '%s' (%s). Refitting...", model_file, e)
                    predictor.fit()
                    predictor.save_model(model_file)
            else:
                logger.info("XGBoost artifact not found at '%s'. Fitting fresh model...", model_file)
                predictor.fit()
                predictor.save_model(model_file)

            self._risk_predictor = predictor
        return self._risk_predictor

    def get_safe_career_pivots(
        self,
        soc_code: str,
        current_threat: Optional[float] = None,
        top_k: int = 5,
    ) -> List[SafeTransitionRole]:
        """
        Orchestrates retrieving safe career transition options within candidate's capability cluster.
        Fetches current threat score from IOnetRepository if not provided.

        Args:
            soc_code: Source 6-digit SOC occupation code (e.g., '15-1252.00')
            current_threat: Optional baseline threat score
            top_k: Maximum number of safe transition recommendations

        Returns:
            List[SafeTransitionRole]: Ordered list of safe target occupations
        """
        logger.info("Retrieving safe career transition pivots for SOC '%s' (top_k=%d)...", soc_code, top_k)
        start_time = time.perf_counter()

        if current_threat is None:
            risk = self.onet_repo.get_risk_profile(soc_code)
            if risk is not None:
                current_threat = risk.threat_score
            else:
                logger.warning("SOC code '%s' not found in repository. Assuming baseline threat 50.0.", soc_code)
                current_threat = 50.0

        pivots = self.clustering_engine.get_safe_pivots(
            soc_code=soc_code,
            current_threat=current_threat,
            top_k=top_k,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Found %d safe pivots for SOC '%s' in %.2f ms.", len(pivots), soc_code, elapsed_ms)
        return pivots

    def predict_hybrid_role_risk(self, profile: CandidateProfile) -> RiskProfile:
        """
        Infers AI automation risk profile for custom, cross-functional, or non-standard candidate roles.

        Args:
            profile: CandidateProfile entity extracted from user's resume

        Returns:
            RiskProfile: Multi-dimensional risk score, level, and Top 3 defensive/vulnerable skills
        """
        logger.info("Predicting AI automation risk profile for candidate '%s'...", profile.full_name or "Unknown")
        return self.risk_predictor.predict_risk(profile)

    def analyze_candidate_transition(
        self,
        profile: CandidateProfile,
        current_soc_code: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        End-to-End Career Advisory Workflow Orchestration:
        1. Predicts candidate risk profile (threat score, defense skills, vulnerabilities).
        2. If SOC code is missing, matches candidate title using IOnetRepository.
        3. Retrieves safe transition pivot roles using IClusteringEngine.
        4. Packages unified domain payload ready for LangGraph Copilot and Presentation layers.

        Args:
            profile: CandidateProfile entity
            current_soc_code: Optional matched SOC code
            top_k: Number of safe transition roles to find

        Returns:
            Dict containing unified risk profile, safe pivots, and metadata.
        """
        start_total = time.perf_counter()
        logger.info("Analyzing comprehensive transition options for candidate '%s'...", profile.full_name or "Unknown")

        # Step 1: Predict Risk Profile via XGBoost & SHAP
        risk_profile = self.predict_hybrid_role_risk(profile)

        # Step 2: Resolve SOC code if absent
        if not current_soc_code and profile.current_title:
            matched_soc, _, conf = self.onet_repo.match_job_title(profile.current_title)
            if conf >= settings.ROUTER_CONFIDENCE_THRESHOLD:
                current_soc_code = matched_soc

        # Step 3: Retrieve Safe Pivots via Clustering
        safe_pivots: List[SafeTransitionRole] = []
        if current_soc_code:
            try:
                safe_pivots = self.get_safe_career_pivots(
                    soc_code=current_soc_code,
                    current_threat=risk_profile.threat_score,
                    top_k=top_k,
                )
            except Exception as e:
                logger.warning("Could not find safe pivots for SOC '%s': %s", current_soc_code, e)

        total_elapsed_ms = (time.perf_counter() - start_total) * 1000

        result = {
            "candidate_name": profile.full_name,
            "current_title": profile.current_title,
            "matched_soc_code": current_soc_code,
            "risk_profile": risk_profile,
            "safe_transition_roles": safe_pivots,
            "top_defense_skills": risk_profile.defense_skills,
            "analysis_latency_ms": round(total_elapsed_ms, 2),
        }

        logger.info(
            "Candidate transition analysis completed in %.2f ms (Risk: %.1f, Safe Pivots: %d)",
            total_elapsed_ms,
            risk_profile.threat_score,
            len(safe_pivots),
        )
        return result
