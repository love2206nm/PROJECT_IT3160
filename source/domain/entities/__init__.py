"""
Domain Entities Package.
Contains all core business entities, value objects, and schemas of the system.
Independent of external frameworks, databases, or parsers.
"""

from source.domain.entities.candidate import (
    SkillItem,
    WorkExperience,
    CandidateProfile,
)
from source.domain.entities.risk import (
    RiskLevel,
    SubRiskScores,
    RiskProfile,
)
from source.domain.entities.transition import (
    SafeTransitionRole,
    SkillGap,
)
from source.domain.entities.roadmap import (
    RoadmapPhase,
    CopilotRoadmap,
)
from source.domain.entities.analysis import (
    AnalysisResult,
)
from source.domain.entities.role import (
    OnetRole,
)
from source.domain.entities.prompt import (
    CopilotPromptCatalog,
)

__all__ = [
    # Candidate entities
    "SkillItem",
    "WorkExperience",
    "CandidateProfile",
    # Role entity
    "OnetRole",
    # Risk entities
    "RiskLevel",
    "SubRiskScores",
    "RiskProfile",
    # Transition entities
    "SafeTransitionRole",
    "SkillGap",
    # Roadmap entities
    "RoadmapPhase",
    "CopilotRoadmap",
    # Aggregated Analysis result
    "AnalysisResult",
    # Prompt Catalog
    "CopilotPromptCatalog",
]


