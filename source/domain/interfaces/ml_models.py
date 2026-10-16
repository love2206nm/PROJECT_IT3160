"""
Machine Learning Core Domain Protocols (Ports).
Defines abstract contracts for Agglomerative Clustering and XGBoost Risk Predictor.
"""

from typing import Protocol, List, runtime_checkable
from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.risk import RiskProfile
from source.domain.entities.transition import SafeTransitionRole


@runtime_checkable
class IClusteringEngine(Protocol):
    """Protocol for finding safe career transition pivots in capability space."""

    def get_safe_pivots(
        self, soc_code: str, current_threat: float, top_k: int = 5
    ) -> List[SafeTransitionRole]:
        """
        Find top safe career pivots within the candidate's skill cluster.

        Args:
            soc_code: Current 6-digit SOC code of candidate
            current_threat: Current threat score of candidate's role
            top_k: Number of safe pivot roles to recommend

        Returns:
            List[SafeTransitionRole]: Ranked list of safe target occupations
        """
        ...


@runtime_checkable
class IRiskPredictor(Protocol):
    """Protocol for predicting AI automation risk score from candidate profile."""

    def predict_risk(self, profile: CandidateProfile) -> RiskProfile:
        """
        Infer threat score and multi-dimensional risk profile for custom/non-standard roles.

        Args:
            profile: Extracted candidate profile

        Returns:
            RiskProfile: Inferred domain risk profile
        """
        ...
