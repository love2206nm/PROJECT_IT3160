"""
CORS (Cross-Origin Resource Sharing) Middleware Setup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from source.infrastructure.config import settings


def setup_cors(app: FastAPI) -> None:
    """Configures CORS middleware on the FastAPI application."""
    origins = settings.CORS_ORIGINS if settings.CORS_ORIGINS else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
