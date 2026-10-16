"""
Data Layer Configuration Schemas.
Defines injectable configuration models for raw datasets and clean data storage.
Eliminates hardcoded paths across data loaders and repositories.
"""

from pathlib import Path
from pydantic import BaseModel, Field
from source.infrastructure.config import settings


class RawDataConfig(BaseModel):
    """
    Configuration for all raw O*NET data file locations and filenames.
    Defaults to system settings but can be overridden 100% via dependency injection.
    """
    raw_dir: Path = Field(default_factory=lambda: settings.DATA_RAW_DIR)
    crosswalk_file: str = "soc-crosswalk.csv"
    skills_file: str = "essential_skills.csv"
    knowledge_file: str = "knowledge.csv"
    abilities_file: str = "abilities.csv"
    software_skills_file: str = "software_skills.csv"
    task_ratings_file: str = "task_ratings.csv"
    salaries_file: str = "voiced-real-pay-by-metro-2025.csv"
    threat_index_file: str = "ai-career-threat-index.csv"
    job_titles_file: str = "job_titles.csv"

    @property
    def crosswalk_path(self) -> Path:
        return Path(self.raw_dir) / self.crosswalk_file

    @property
    def skills_path(self) -> Path:
        return Path(self.raw_dir) / self.skills_file

    @property
    def knowledge_path(self) -> Path:
        return Path(self.raw_dir) / self.knowledge_file

    @property
    def abilities_path(self) -> Path:
        return Path(self.raw_dir) / self.abilities_file

    @property
    def software_skills_path(self) -> Path:
        return Path(self.raw_dir) / self.software_skills_file

    @property
    def task_ratings_path(self) -> Path:
        return Path(self.raw_dir) / self.task_ratings_file

    @property
    def salaries_path(self) -> Path:
        return Path(self.raw_dir) / self.salaries_file

    @property
    def threat_index_path(self) -> Path:
        return Path(self.raw_dir) / self.threat_index_file

    @property
    def job_titles_path(self) -> Path:
        return Path(self.raw_dir) / self.job_titles_file


class CleanDataConfig(BaseModel):
    """
    Configuration for clean data store paths and output artifact filenames.
    Defines partition directories for initial (unpruned) and final (model-ready) data.
    """
    clean_dir: Path = Field(default_factory=lambda: settings.DATA_CLEAN_DIR)
    initial_subdir: str = "initial"
    final_subdir: str = "final"

    feature_matrix_full_file: str = "soc_feature_matrix_full.parquet"
    feature_matrix_gold_file: str = "soc_feature_matrix.parquet"
    metadata_file: str = "soc_metadata.parquet"
    rag_corpus_file: str = "rag_corpus.json"
    pruning_report_file: str = "pruning_report.json"

    @property
    def initial_dir(self) -> Path:
        return Path(self.clean_dir) / self.initial_subdir

    @property
    def final_dir(self) -> Path:
        return Path(self.clean_dir) / self.final_subdir

    @property
    def initial_feature_matrix_path(self) -> Path:
        return self.initial_dir / self.feature_matrix_full_file

    @property
    def final_feature_matrix_path(self) -> Path:
        return self.final_dir / self.feature_matrix_gold_file

    @property
    def final_metadata_path(self) -> Path:
        return self.final_dir / self.metadata_file

    @property
    def final_rag_corpus_path(self) -> Path:
        return self.final_dir / self.rag_corpus_file

    @property
    def final_pruning_report_path(self) -> Path:
        return self.final_dir / self.pruning_report_file
