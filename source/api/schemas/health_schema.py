"""
Pydantic Schemas for System Health Monitoring.
"""

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """System health check payload for liveness and readiness probes."""
    status: str = Field("healthy", description="Application operational status")
    app_name: str = Field("AI Career Copilot", description="Name of the service")
    app_env: str = Field("development", description="Environment stage")
    debug: bool = Field(False, description="Debug mode flag")
    version: str = Field("1.0.0", description="API version")
