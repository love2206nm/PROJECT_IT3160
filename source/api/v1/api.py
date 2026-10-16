"""
API v1 Router Aggregator.
Combines CV Ingestion, Career Analysis, Copilot Chat, and System Health endpoints.
"""

from fastapi import APIRouter

from source.api.v1.endpoints import cv, analysis, copilot, health

api_router = APIRouter()

api_router.include_router(health.router, tags=["System Health"])
api_router.include_router(cv.router, prefix="/cv", tags=["CV Analysis & HITL"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["Career Analysis"])
api_router.include_router(copilot.router, prefix="/copilot", tags=["Interactive Copilot"])
