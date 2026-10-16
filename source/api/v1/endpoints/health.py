"""
Health check endpoints for container orchestrators and status monitoring.
"""

from fastapi import APIRouter
from source.api.schemas.health_schema import HealthCheckResponse
from source.infrastructure.config import settings

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="System Health & Readiness Check",
    tags=["System"],
)
async def get_health_status() -> HealthCheckResponse:
    """Returns application status for Docker/K8s liveness probes."""
    return HealthCheckResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        app_env=settings.APP_ENV,
        debug=settings.DEBUG,
        version="1.0.0",
    )
