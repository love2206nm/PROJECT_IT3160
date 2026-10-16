"""
Infrastructure Configuration Module.
Loads environment variables and application settings using pydantic-settings.
"""

from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration management for AI Career Copilot.
    Validates environment variables and provides sensible defaults.
    """
    # Application Info
    APP_NAME: str = "AI Career Copilot"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    # Project Directories (Defaults to absolute workspace paths)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_RAW_DIR: Path = Field(default_factory=lambda: Path("data/raw_data"))
    DATA_CLEAN_DIR: Path = Field(default_factory=lambda: Path("data/clean_data"))
    MODEL_DIR: Path = Field(default_factory=lambda: Path("models"))
    FAISS_INDEX_DIR: Path = Field(default_factory=lambda: Path("models/faiss_index"))
    STATIC_DIR: Path = Field(default_factory=lambda: Path("frontend"))

    # LLM Providers & API Keys
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    FAST_LLM_PROVIDER: str = "deepseek"  # "deepseek" | "gemini" | "openai"
    FAST_LLM_MODEL: str = "deepseek-chat"
    REASONING_LLM_MODEL: str = "gemini-1.5-pro"

    # Hybrid Router & ML Algorithm Parameters
    ROUTER_CONFIDENCE_THRESHOLD: float = 0.90
    CLUSTERING_N_CLUSTERS: int = 23
    CORRELATION_THRESHOLD: float = 0.85
    CROSS_GROUP_THRESHOLD: float = 0.95

    # XGBoost Risk Predictor Hyperparameters
    XGB_MAX_DEPTH: int = 3
    XGB_LEARNING_RATE: float = 0.05
    XGB_N_ESTIMATORS: int = 100
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.8
    XGB_REG_ALPHA: float = 1.0
    XGB_REG_LAMBDA: float = 2.0

    # Transition Utility Weights
    WEIGHT_SIMILARITY: float = 0.50
    WEIGHT_AI_SAFETY: float = 0.35
    WEIGHT_SALARY: float = 0.15
    SAFETY_MARGIN_DELTA: float = 10.0

    # RAG & Copilot Parameters
    RAG_TOP_K: int = 5
    RAG_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RAG_USE_RERANKER: bool = True
    MAX_CRITIC_ITERATIONS: int = 2

    # Logging Parameters
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Field(default_factory=lambda: Path("logs"))
    LOG_FILE: str = "job_advisor.log"
    LOG_TO_CONSOLE: bool = True
    LOG_TO_FILE: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
