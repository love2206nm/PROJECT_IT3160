"""
Application Server Runner.
Entrypoint to start the AI Career Copilot FastAPI server locally or in production.
"""

import os
import uvicorn

from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger
from source.api.app import app  # Export app for ASGI servers like gunicorn/uvicorn

logger = get_logger(__name__)

if __name__ == "__main__":
    # Support dynamic PORT environment variable (standard in Hugging Face, Render, Railway, Heroku)
    port = int(os.getenv("PORT", settings.API_PORT))
    host = os.getenv("HOST", settings.API_HOST)

    logger.info("Starting %s on http://%s:%s", settings.APP_NAME, host, port)
    logger.info("Interactive Swagger API Docs: http://%s:%s/docs", host, port)

    uvicorn.run(
        "source.api.app:app",
        host=host,
        port=port,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
