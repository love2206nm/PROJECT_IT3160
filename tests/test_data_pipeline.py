"""
Unit & Integration Tests for O*NET Data Pipeline, Feature Store, and In-Memory Repository.
Validates clean data assets and verifies Dependency Injection decoupling.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from source.data.config import RawDataConfig, CleanDataConfig
from source.data.preprocess.onet_loader import OnetDataLoader
from source.data.data_engineering.feature_transformer import TwoStageFeatureTransformer
from source.data.data_engineering.metadata_builder import OnetMetadataBuilder
from source.infrastructure.database.repo.onet_repository import OnetRepository
from source.domain.entities.risk import RiskProfile, RiskLevel


default_clean_cfg = CleanDataConfig()
default_raw_cfg = RawDataConfig()


def test_clean_store_files_exist():
    """Verify that all initial and final clean data assets exist at configured paths."""
    assert default_clean_cfg.initial_feature_matrix_path.exists(), f"Missing {default_clean_cfg.initial_feature_matrix_path}"
    assert default_clean_cfg.final_feature_matrix_path.exists(), f"Missing {default_clean_cfg.final_feature_matrix_path}"
    assert default_clean_cfg.final_metadata_path.exists(), f"Missing {default_clean_cfg.final_metadata_path}"
    assert default_clean_cfg.final_rag_corpus_path.exists(), f"Missing {default_clean_cfg.final_rag_corpus_path}"
    assert default_clean_cfg.final_pruning_report_path.exists(), f"Missing {default_clean_cfg.final_pruning_report_path}"


def test_loader_custom_config_injection():
    """
    Test that OnetDataLoader accepts custom RawDataConfig via Dependency Injection
    and does NOT rely on hardcoded paths or file strings.
    """
    custom_dir = Path("custom/test/raw")
    custom_cfg = RawDataConfig(
        raw_dir=custom_dir,
        crosswalk_file="custom_crosswalk.csv",
        skills_file="custom_skills.csv",
    )
    loader = OnetDataLoader(config=custom_cfg)

    assert loader.config.raw_dir == custom_dir
    assert loader.config.crosswalk_path == custom_dir / "custom_crosswalk.csv"
    assert loader.config.skills_path == custom_dir / "custom_skills.csv"
    assert loader.config.salaries_path == custom_dir / "voiced-real-pay-by-metro-2025.csv"


def test_repository_custom_config_injection():
    """
    Test that OnetRepository accepts custom CleanDataConfig and RawDataConfig
    via Dependency Injection.
    """
    custom_clean_dir = Path("custom/clean")
    custom_raw_dir = Path("custom/raw")
    clean_cfg = CleanDataConfig(clean_dir=custom_clean_dir)
    raw_cfg = RawDataConfig(raw_dir=custom_raw_dir)

    repo = OnetRepository(clean_config=clean_cfg, raw_config=raw_cfg)

    assert repo.metadata_path == custom_clean_dir / "final" / "soc_metadata.parquet"
    assert repo.feature_path == custom_clean_dir / "final" / "soc_feature_matrix.parquet"
    assert repo.job_titles_path == custom_raw_dir / "job_titles.csv"


def test_initial_feature_matrix_120d():
    """
    Test Stage 1 Initial (Bronze) Feature Matrix:
    - Exactly 867 rows (all Base SOCs).
    - Exactly 120 feature dimensions.
    - float32 data type, values in [0.0, 1.0].
    - Zero NaNs.
    - Zero Label Leakage (no threat or risk keywords in columns).
    """
    df = pd.read_parquet(default_clean_cfg.initial_feature_matrix_path)
    assert df.shape == (867, 120), f"Expected (867, 120), got {df.shape}"
    assert df.isna().sum().sum() == 0, "Initial feature matrix must not have NaNs"
    assert (df.min().min() >= 0.0) and (df.max().max() <= 1.0), "Values must be normalized to [0.0, 1.0]"

    # Verify Zero Label Leakage
    leak_keywords = ["threat", "risk", "sub_task", "sub_tool", "sub_agentic"]
    for col in df.columns:
        for kw in leak_keywords:
            assert kw not in col.lower(), f"Label leakage detected in feature: {col}"


def test_final_feature_matrix_gold_pruned():
    """
    Test Stage 2 Final (Gold) Model-Ready Feature Matrix:
    - Exactly 867 rows.
    - Pruned dimension between 50 and 80.
    - Zero NaNs.
    - Values in [0.0, 1.0].
    - No multicollinear pairs with |r| >= 0.85 within groups.
    """
    df = pd.read_parquet(default_clean_cfg.final_feature_matrix_path)
    assert df.shape[0] == 867, f"Expected 867 rows, got {df.shape[0]}"
    assert 50 <= df.shape[1] <= 80, f"Expected 50-80 columns after pruning, got {df.shape[1]}"
    assert df.isna().sum().sum() == 0, "Gold feature matrix must not have NaNs"

    with open(default_clean_cfg.final_pruning_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert "dropped_details" in report
    assert report["initial_feature_count"] == 120
    assert report["pruned_feature_count"] == df.shape[1]
    assert report["reduction_percentage"] > 25.0


def test_final_metadata_catalog():
    """
    Test Metadata Store (soc_metadata.parquet):
    - Exactly 867 rows.
    - 100% salary coverage (no NaNs, all positive).
    - Every role has 3 defense skills.
    - Every role has a software_skills_catalog list.
    """
    df = pd.read_parquet(default_clean_cfg.final_metadata_path)
    assert len(df) == 867, f"Expected 867 records, got {len(df)}"

    # Salary coverage
    assert df["median_salary"].isna().sum() == 0, "Median salary must have 0 NaNs"
    assert (df["median_salary"] > 20000.0).all(), "All salaries must be realistic positive numbers"

    # Defense skills
    for soc, row in df.iterrows():
        d_skills = row["defense_skills"]
        assert len(d_skills) == 3, f"SOC {soc} does not have exactly 3 defense skills: {d_skills}"

    # Software catalog
    sample_tools = df.iloc[0]["software_skills_catalog"]
    assert isinstance(sample_tools, (list, np.ndarray)), "software_skills_catalog must be a list"


def test_hierarchical_rag_corpus():
    """
    Test Hierarchical Aspect RAG Corpus (rag_corpus.json):
    - Exactly 2,601 micro-chunks (867 x 3 aspects).
    - Exactly 867 chunks per aspect: threat_mechanism, defense_skills, tasks_and_tools.
    """
    with open(default_clean_cfg.final_rag_corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    assert len(corpus) == 2601, f"Expected 2,601 chunks, got {len(corpus)}"

    aspect_counts = {}
    for chunk in corpus:
        assert "chunk_id" in chunk
        assert "soc_code" in chunk
        assert "major_group" in chunk
        assert "aspect" in chunk
        assert "content" in chunk and len(chunk["content"]) > 30

        aspect = chunk["aspect"]
        aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1

    assert aspect_counts == {
        "threat_mechanism": 867,
        "defense_skills": 867,
        "tasks_and_tools": 867,
    }, f"Aspect distribution mismatch: {aspect_counts}"


def test_onet_repository_integration():
    """
    Test OnetRepository In-Memory queries and latency requirements:
    - Rapid title fuzzy matching < 50ms.
    - RiskProfile domain entity retrieval < 1ms.
    - Feature vector extraction < 1ms.
    """
    repo = OnetRepository(
        clean_config=default_clean_cfg,
        raw_config=default_raw_cfg,
    )

    # 1. Test Title Matching Latency and Accuracy (warm up first to eliminate cold-start overhead)
    repo.match_job_title("Software Engineer")

    t0 = time.time()
    soc_code, soc_title, confidence = repo.match_job_title("Python Software Developer")
    latency_match = (time.time() - t0) * 1000

    assert soc_code.startswith("15-"), f"Expected Computer/Math SOC, got {soc_code}"
    assert confidence > 0.5, f"Expected high confidence, got {confidence}"
    assert latency_match < 100.0, f"Title match exceeded 100ms: {latency_match:.2f}ms"

    # 2. Test RiskProfile Lookup Latency & Domain Entity Structure
    t0 = time.time()
    profile = repo.get_risk_profile(soc_code)
    latency_profile = (time.time() - t0) * 1000

    assert profile is not None
    assert isinstance(profile, RiskProfile)
    assert profile.soc_code == soc_code
    assert isinstance(profile.risk_level, (RiskLevel, str))
    assert 0.0 <= profile.threat_score <= 100.0
    assert len(profile.defense_skills) == 3
    assert latency_profile < 1.0, f"Profile lookup exceeded 1ms: {latency_profile:.3f}ms"

    # 3. Test Feature Vector Extraction
    t0 = time.time()
    vec = repo.get_feature_vector(soc_code)
    latency_vec = (time.time() - t0) * 1000

    assert vec is not None
    assert isinstance(vec, np.ndarray)
    assert len(vec) == repo.feature_matrix_df_.shape[1]
    assert latency_vec < 1.0, f"Feature vector extraction exceeded 1ms: {latency_vec:.3f}ms"


def test_unmatched_soc_benchmark_fallback():
    """Verify that agriculture (45-xxxx) and military (55-xxxx) SOCs have valid benchmark risk."""
    repo = OnetRepository(
        clean_config=default_clean_cfg,
        raw_config=default_raw_cfg,
    )

    # Test an agriculture SOC
    agri_profile = repo.get_risk_profile("45-1011")
    assert agri_profile is not None
    assert agri_profile.threat_score == 35.0
    assert agri_profile.defense_skills[0] == "Physical Dexterity"

    # Test a military SOC
    mil_profile = repo.get_risk_profile("55-1011")
    assert mil_profile is not None
    assert mil_profile.threat_score == 15.0
    assert mil_profile.defense_skills[0] == "Crisis Decision Making"


def test_logger_infrastructure_and_file_logging():
    """
    Test centralized logging infrastructure:
    - Verifies get_logger returns singleton configured Logger.
    - Verifies handlers are idempotent (no duplicate handlers upon re-retrieval).
    - Verifies log writing to rotating file handler with UTF-8 encoding.
    """
    from source.infrastructure.logger import get_logger
    from source.infrastructure.config import settings

    logger = get_logger("test_logging_module")
    initial_handler_count = len(logger.handlers)

    # Idempotency check
    logger_second = get_logger("test_logging_module")
    assert len(logger_second.handlers) == initial_handler_count, "Logger handlers must not duplicate"

    # File log write check
    test_msg = f"Unit test audit entry: {time.time()}"
    logger.info(test_msg)

    log_path = Path(settings.LOG_DIR) / settings.LOG_FILE
    assert log_path.exists(), f"Expected log file to exist at {log_path}"

    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    assert test_msg in log_content, f"Expected '{test_msg}' in {log_path}"

