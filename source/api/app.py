"""
FastAPI Application Entrypoint & Server Configuration.
Integrates Routers, Middlewares, CORS, Exception Handlers, and Static Frontend Mount.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from source.api.v1.api import api_router
from source.api.v1.endpoints.health import router as health_router
from source.api.middleware.cors import setup_cors
from source.api.middleware.error_handler import setup_exception_handlers
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event manager for startup and shutdown procedures."""
    logger.info("Initializing %s in [%s] mode...", settings.APP_NAME, settings.APP_ENV)
    yield
    logger.info("Shutting down %s...", settings.APP_NAME)


def create_app() -> FastAPI:
    """Application factory initializing and configuring the FastAPI app."""
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "AI Recruiter & Career Copilot API. "
            "Analyzes CVs, predicts AI automation threat using XGBoost & O*NET catalog, "
            "discovers safe career transitions via Agglomerative Clustering, and generates "
            "actionable 30-60-90 days roadmaps with LangGraph State Machine & Critic Loop."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. Setup CORS & Global Exception Handlers
    setup_cors(app)
    setup_exception_handlers(app)

    # 2. Register Root Health Check & API v1 Routers
    app.include_router(health_router, tags=["Health Probe"])
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # 3. Mount Static Frontend (All-in-One Deployment Mode)
    static_dir = settings.STATIC_DIR
    if static_dir.exists() and (static_dir / "index.html").exists():
        logger.info("Mounting Static Frontend from directory: %s", static_dir.resolve())
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
    else:
        logger.debug("Static frontend directory '%s' not found or empty. API-only mode active.", static_dir)

    return app


app = create_app()
