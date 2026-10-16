"""
O*NET Repository Domain Protocol (Interface).
Defines abstract contracts for fast in-memory O*NET queries, title matching, and feature vectors.
"""

from typing import Protocol, Optional, Tuple, Dict, Any, List, runtime_checkable
import numpy as np
from source.domain.entities.risk import RiskProfile


@runtime_checkable
class IOnetRepository(Protocol):
    """Protocol for in-memory O*NET repository supporting 0ms lookups and rapid title matching."""

    def match_job_title(self, raw_title: Optional[str]) -> Tuple[Optional[str], Optional[str], float]:
        """
        Fuzzy match candidate job title against 54k O*NET market titles.
        Returns:
            Tuple[Optional[str], Optional[str], float]: (matched_soc_code, official_soc_title, confidence_score)
            or (None, None, 0.0) if unmatchable.
        """
        ...

    def match_role_by_skills(self, skills: List[str]) -> Tuple[Optional[str], Optional[str], float]:
        """
        Calculates maximum skill overlap against 867 O*NET occupations' software_skills_catalog and hot_tech.
        Returns:
            Tuple[Optional[str], Optional[str], float]: (matched_soc, matched_title, overlap_ratio)
            or (None, None, 0.0) if no overlapping skills found.
        """
        ...

    def match_role_by_education(self, major: str) -> Tuple[Optional[str], Optional[str], float]:
        """
        Maps academic major (e.g., Computer Science, Information Technology, Economics, Finance, Marketing)
        to the most relevant foundational entry-level O*NET SOC occupation.
        Returns:
            Tuple[Optional[str], Optional[str], float]: (matched_soc, matched_title, confidence)
        """
        ...

    def get_risk_profile(self, soc_code: str) -> Optional[RiskProfile]:
        """
        Retrieve domain RiskProfile entity for a given SOC code in <1ms.
        Returns:
            Optional[RiskProfile]: Validated domain RiskProfile or None if not found.
        """
        ...

    def get_feature_vector(self, soc_code: str) -> Optional[np.ndarray]:
        """
        Retrieve normalized model-ready feature vector (~76D) for machine learning models.
        Returns:
            Optional[np.ndarray]: 1D float32 numpy array or None if not found.
        """
        ...

    def get_metadata(self, soc_code: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve raw metadata dictionary for a given SOC code.
        Returns:
            Optional[Dict[str, Any]]: Metadata record or None if not found.
        """
        ...

    def resolve_candidate_soc(self, profile: Any) -> Tuple[Optional[str], Optional[str], float, bool]:
        """
        Unified 5-tier SOC occupation discovery for candidate profiles:
        1. Job Title exact/fuzzy match against 54k market titles (skipping generic student keywords).
        2. Skill overlap match against 867 O*NET occupations' software skills and hot technologies.
        3. Education / academic major match to foundational entry-level O*NET occupation.

        Returns:
            Tuple[Optional[str], Optional[str], float, bool]:
            (soc_code, soc_title, confidence, is_student_or_fresher)
        """
        ...

