"""
Domain Layer - Core Business Logic and Contracts.
"""

from source.domain.entities import (
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
    CopilotPromptCatalog,
)
from source.domain.interfaces import (
    IDataLoader,
    IFeatureTransformer,
    IMetadataBuilder,
    IOnetRepository,
    IDocumentParser,
    IProfileExtractor,
    IClusteringEngine,
    IRiskPredictor,
    IEvidenceRetriever,
)
from source.domain.exceptions import (
    DomainError,
    CVExtractionError,
    TitleMatchingError,
    RoleNotFoundError,
    ModelInferenceError,
    RoadmapGenerationError,
)

__all__ = [
    # Entities
    "SkillItem",
    "WorkExperience",
    "CandidateProfile",
    "OnetRole",
    "RiskLevel",
    "SubRiskScores",
    "RiskProfile",
    "SafeTransitionRole",
    "SkillGap",
    "RoadmapPhase",
    "CopilotRoadmap",
    "AnalysisResult",
    "CopilotPromptCatalog",

    # Interfaces
    "IDataLoader",
    "IFeatureTransformer",
    "IMetadataBuilder",
    "IOnetRepository",
    "IDocumentParser",
    "IProfileExtractor",
    "IClusteringEngine",
    "IRiskPredictor",
    "IEvidenceRetriever",
    # Exceptions
    "DomainError",
    "CVExtractionError",
    "TitleMatchingError",
    "RoleNotFoundError",
    "ModelInferenceError",
    "RoadmapGenerationError",
]

