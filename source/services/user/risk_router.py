"""
Hybrid AI Automation Risk Routing Strategy (Use Case Layer).
Selects between Deterministic O*NET Lookup (0ms exact ground truth)
and XGBoost Machine Learning Regressor (inferred custom role risk)
based on confidence threshold (default >= 0.90) or Human-in-the-Loop user confirmation.
"""

from typing import Optional
import time

from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.risk import RiskProfile
from source.domain.interfaces.onet_repository import IOnetRepository
from source.domain.interfaces.ml_models import IRiskPredictor
from source.infrastructure.database.repo.onet_repository import OnetRepository
from source.model.ML_model.risk_predictor import XGBoostRiskPredictor
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class HybridRiskRouter:
    """
    Application Strategy orchestrating AI automation risk determination.
    Strictly adheres to Clean Architecture via IOnetRepository and IRiskPredictor protocols.
    """

    def __init__(
        self,
        onet_repo: Optional[IOnetRepository] = None,
        risk_predictor: Optional[IRiskPredictor] = None,
        threshold: Optional[float] = None,
    ):
        """
        Initializes HybridRiskRouter with dependency injection.

        Args:
            onet_repo: Injected IOnetRepository protocol implementation (lazy-loaded if None)
            risk_predictor: Injected IRiskPredictor protocol implementation (lazy-loaded if None)
            threshold: Confidence threshold for deterministic lookup (defaults to settings.ROUTER_CONFIDENCE_THRESHOLD)
        """
        self._onet_repo = onet_repo
        self._risk_predictor = risk_predictor
        self.threshold = threshold if threshold is not None else settings.ROUTER_CONFIDENCE_THRESHOLD

        logger.debug(
            "HybridRiskRouter initialized (threshold=%.2f, repo=%s, predictor=%s)",
            self.threshold,
            type(onet_repo).__name__ if onet_repo else "Lazy",
            type(risk_predictor).__name__ if risk_predictor else "Lazy",
        )

    @property
    def onet_repo(self) -> IOnetRepository:
        """Lazy-loads OnetRepository if not provided."""
        if self._onet_repo is None:
            self._onet_repo = OnetRepository()
        return self._onet_repo

    @property
    def risk_predictor(self) -> IRiskPredictor:
        """Lazy-loads XGBoostRiskPredictor if not provided."""
        if self._risk_predictor is None:
            predictor = XGBoostRiskPredictor(
                model_path=settings.MODEL_DIR / "xgb_risk_model.joblib"
            )
            model_file = settings.MODEL_DIR / "xgb_risk_model.joblib"
            if model_file.exists():
                try:
                    predictor.load_model(model_file)
                except Exception as e:
                    logger.warning("Could not load XGBoost model from '%s' (%s). Refitting...", model_file, e)
                    predictor.fit()
                    predictor.save_model(model_file)
            else:
                logger.info("XGBoost artifact not found at '%s'. Fitting fresh model...", model_file)
                predictor.fit()
                predictor.save_model(model_file)
            self._risk_predictor = predictor
        return self._risk_predictor

    def determine_risk(
        self,
        profile: CandidateProfile,
        soc_code: Optional[str] = None,
        confidence: float = 0.0,
        user_confirmed: bool = False,
    ) -> RiskProfile:
        """
        Routes and determines the risk profile for a candidate.

        Branching logic:
        - If (confidence >= threshold OR user_confirmed is True) AND soc_code is valid:
          -> Deterministic 0ms lookup from IOnetRepository.
        - Otherwise (custom, hybrid, or unmatched role):
          -> ML inference via IRiskPredictor (XGBoost + SHAP).

        Args:
            profile: CandidateProfile domain entity
            soc_code: Matched O*NET SOC occupation code if available
            confidence: Title matching confidence score [0.0, 1.0]
            user_confirmed: True if human-in-the-loop validated the SOC code

        Returns:
            RiskProfile: Validated domain entity
        """
        start_time = time.perf_counter()
        is_deterministic = (confidence >= self.threshold or user_confirmed) and bool(soc_code)

        if is_deterministic and soc_code:
            logger.info(
                "[RiskRouter] Deterministic lookup selected for SOC '%s' (conf=%.3f, threshold=%.2f, confirmed=%s)",
                soc_code,
                confidence,
                self.threshold,
                user_confirmed,
            )
            risk_profile = self.onet_repo.get_risk_profile(soc_code)
            if risk_profile is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                level_str = getattr(risk_profile.risk_level, "value", str(risk_profile.risk_level))
                logger.info(
                    "[RiskRouter] Deterministic lookup completed in %.2f ms (Score: %.1f, Level: %s)",
                    elapsed_ms,
                    risk_profile.threat_score,
                    level_str,
                )
                return risk_profile

            logger.warning(
                "[RiskRouter] SOC '%s' not found in O*NET repository. Falling back to ML predictor.",
                soc_code,
            )

        logger.info(
            "[RiskRouter] ML Risk Predictor selected for candidate '%s' (conf=%.3f < threshold=%.2f, SOC=%s)",
            profile.full_name or "Unknown",
            confidence,
            self.threshold,
            soc_code,
        )
        risk_profile = self.risk_predictor.predict_risk(profile)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        level_str = getattr(risk_profile.risk_level, "value", str(risk_profile.risk_level))
        logger.info(
            "[RiskRouter] ML prediction completed in %.2f ms (Score: %.1f, Level: %s)",
            elapsed_ms,
            risk_profile.threat_score,
            level_str,
        )
        return risk_profile
