"""
Global Exception Handlers for FastAPI Application.
Transforms Domain/Infrastructure exceptions into consistent JSON error responses.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from source.domain.exceptions import CVExtractionError, DomainError
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Registers centralized exception handlers."""

    @app.exception_handler(CVExtractionError)
    async def cv_extraction_error_handler(request: Request, exc: CVExtractionError) -> JSONResponse:
        logger.warning("CVExtractionError on %s: %s", request.url.path, str(exc))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "CVExtractionError",
                "message": str(exc),
                "detail": "Failed to parse or extract structured data from resume.",
            },
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
        logger.warning("KeyError (Session not found) on %s: %s", request.url.path, str(exc))
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "SessionNotFound",
                "message": str(exc).strip("'\""),
                "detail": "Requested analysis session does not exist or has expired.",
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning("ValueError on %s: %s", request.url.path, str(exc))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "BadRequest",
                "message": str(exc),
                "detail": "Invalid parameters or payload provided.",
            },
        )

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        logger.error("Domain Error on %s: %s", request.url.path, str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": exc.__class__.__name__,
                "message": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled Exception on %s: %s", request.url.path, str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred while processing the request.",
                "detail": str(exc) if getattr(app.state, "debug", False) else None,
            },
        )
