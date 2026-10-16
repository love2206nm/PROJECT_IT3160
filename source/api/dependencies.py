"""
FastAPI Dependency Injection Providers.
Exposes application services and infrastructure components to routers.
"""

from source.services.user.cv_analysis_service import (
    CVAnalysisService,
    get_cv_analysis_service,
)

__all__ = ["CVAnalysisService", "get_cv_analysis_service"]
