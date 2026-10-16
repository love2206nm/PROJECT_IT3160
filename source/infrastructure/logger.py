"""
Centralized Logging Infrastructure Module.
Configures structured, thread-safe console and rotating file logging for the application.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from source.infrastructure.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Factory function to retrieve or configure a standardized logger instance.

    Args:
        name: Module name (typically __name__)

    Returns:
        logging.Logger: Fully configured logger instance
    """
    logger = logging.getLogger(name)

    # Set logging level from centralized settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if the logger is already initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. Console Stream Handler
        if settings.LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 2. Rotating File Handler (10MB per file, 5 backup files)
        if settings.LOG_TO_FILE:
            try:
                log_dir = Path(settings.LOG_DIR)
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / settings.LOG_FILE

                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=10 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception:
                # In restricted container environments, fallback safely to console stream
                pass

        # Do not propagate to root logger to avoid duplicate log lines
        logger.propagate = False

    return logger
