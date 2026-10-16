"""
Domain-Specific Business Exceptions.
Defines explicit, semantically rich exceptions raised within the core domain layer.
Free from presentation (FastAPI HTTP) or database adapter coupling.
"""


class DomainError(Exception):
    """Base exception for all domain and business rule violations."""
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CVExtractionError(DomainError):
    """Raised when resume document cannot be parsed or layout extraction fails."""
    pass


class TitleMatchingError(DomainError):
    """Raised when candidate job title is empty or cannot be matched with acceptable confidence."""
    pass


class RoleNotFoundError(DomainError):
    """Raised when a specified SOC code or occupational title is not found in the O*NET catalog."""
    pass


class ModelInferenceError(DomainError):
    """Raised when ML model (Clustering or XGBoost) fails during inference."""
    pass


class RoadmapGenerationError(DomainError):
    """Raised when Copilot LLM fails to synthesize a valid, evidence-grounded action roadmap."""
    pass
