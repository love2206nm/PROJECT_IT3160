"""
Unit tests for UserCVService (source/services/user/cv_service.py).
Validates orchestration of document parsing and structured profile extraction,
dependency injection, metadata generation, and exception handling.
"""

from pathlib import Path
from typing import Any
import pytest

from source.domain.entities.candidate import CandidateProfile, WorkExperience
from source.domain.exceptions import CVExtractionError
from source.domain.interfaces.parser import IDocumentParser, IProfileExtractor
from source.services.user.cv_service import UserCVService


class MockDocumentParser(IDocumentParser):
    """Mock document parser returning controlled text content."""

    def __init__(self, return_text: str = "Mock Resume Text for Candidate"):
        self.return_text = return_text
        self.called_with: list[Path] = []

    def extract_text(self, file_path: Path) -> str:
        self.called_with.append(file_path)
        if not file_path.exists():
            raise CVExtractionError(f"Mock file not found: {file_path}")
        return self.return_text


class MockProfileExtractor(IProfileExtractor):
    """Mock profile extractor returning controlled CandidateProfile entity."""

    def __init__(self, profile: CandidateProfile | None = None):
        self.profile = profile or CandidateProfile(
            full_name="Le Van B",
            current_title="Senior AI Engineer",
            years_of_experience=3.5,
            hard_skills=["Python", "FastAPI", "PyTorch", "RAG"],
            soft_skills=["Leadership", "Problem Solving"],
            tools_and_technologies=["Docker", "PostgreSQL", "Git"],
            experiences=[
                WorkExperience(
                    company="Tech Innovators",
                    role_title="AI Engineer",
                    duration_months=36,
                    key_responsibilities=["Built RAG pipeline", "Trained models"],
                )
            ],
            education=["B.S. in Computer Science - HUST"],
        )
        self.called_with_text: list[str] = []

    def extract_profile(self, raw_text: str) -> CandidateProfile:
        self.called_with_text.append(raw_text)
        if len(raw_text) < 30:
            raise CVExtractionError("Mock text too short")
        return self.profile


# ==============================================================================
# Tests for UserCVService
# ==============================================================================

def test_user_cv_service_initialization():
    """Verify default and injected initialization."""
    # 1. Injected dependencies
    mock_parser = MockDocumentParser()
    mock_extractor = MockProfileExtractor()
    service = UserCVService(document_parser=mock_parser, profile_extractor=mock_extractor)
    assert service.document_parser is mock_parser
    assert service.profile_extractor is mock_extractor

    # 2. Default initialization
    default_service = UserCVService()
    assert default_service.document_parser is not None
    assert default_service.profile_extractor is not None


def test_extract_raw_text_success(tmp_path: Path):
    """Test extracting clean text from file through service."""
    dummy_file = tmp_path / "resume.txt"
    dummy_file.write_text("Hello World! This is a valid sample resume with enough text.", encoding="utf-8")

    mock_parser = MockDocumentParser(return_text="Extracted text from mock parser")
    service = UserCVService(document_parser=mock_parser)

    result = service.extract_raw_text(dummy_file)
    assert result == "Extracted text from mock parser"
    assert len(mock_parser.called_with) == 1
    assert mock_parser.called_with[0] == dummy_file


def test_extract_raw_text_file_not_found(tmp_path: Path):
    """Test error propagation when file is missing."""
    missing_file = tmp_path / "non_existent.pdf"
    service = UserCVService()

    with pytest.raises(CVExtractionError) as exc_info:
        service.extract_raw_text(missing_file)
    assert "not found" in str(exc_info.value).lower()


def test_extract_profile_from_text_success():
    """Test extracting structured candidate profile from text."""
    sample_text = (
        "Le Van B is a Senior AI Engineer with 3.5 years of experience "
        "specializing in Python, PyTorch, RAG, and FastAPI microservices."
    )
    mock_extractor = MockProfileExtractor()
    service = UserCVService(profile_extractor=mock_extractor)

    profile = service.extract_profile_from_text(sample_text)
    assert profile.full_name == "Le Van B"
    assert profile.current_title == "Senior AI Engineer"
    assert "PyTorch" in profile.hard_skills
    assert len(mock_extractor.called_with_text) == 1


def test_extract_profile_from_text_too_short():
    """Test rejection of empty or excessively short text."""
    service = UserCVService()
    with pytest.raises(CVExtractionError) as exc_info:
        service.extract_profile_from_text("Too short")
    assert "too short" in str(exc_info.value).lower()


def test_process_cv_full_pipeline(tmp_path: Path):
    """Test end-to-end ingestion pipeline: file -> text -> CandidateProfile."""
    dummy_file = tmp_path / "candidate.pdf"
    dummy_file.write_text("Valid content file placeholder", encoding="utf-8")

    mock_parser = MockDocumentParser(return_text="Full resume text with more than thirty characters.")
    mock_extractor = MockProfileExtractor()
    service = UserCVService(document_parser=mock_parser, profile_extractor=mock_extractor)

    profile = service.process_cv(dummy_file)
    assert profile.full_name == "Le Van B"
    assert profile.current_title == "Senior AI Engineer"
    assert len(mock_parser.called_with) == 1
    assert len(mock_extractor.called_with_text) == 1


def test_process_cv_with_metadata(tmp_path: Path):
    """Test ingestion pipeline returning both profile and operational metadata."""
    dummy_file = tmp_path / "candidate_test.docx"
    dummy_file.write_text("A" * 200, encoding="utf-8")

    mock_parser = MockDocumentParser(return_text="This is a detailed mock resume text with over thirty characters.")
    mock_extractor = MockProfileExtractor()
    service = UserCVService(document_parser=mock_parser, profile_extractor=mock_extractor)

    profile, metadata = service.process_cv_with_metadata(dummy_file)

    assert profile.full_name == "Le Van B"
    assert metadata["file_name"] == "candidate_test.docx"
    assert metadata["file_format"] == ".docx"
    assert metadata["raw_characters"] == len("This is a detailed mock resume text with over thirty characters.")
    assert metadata["estimated_tokens"] > 0
    assert "text_extract_latency_ms" in metadata
    assert "profile_extract_latency_ms" in metadata
    assert "total_latency_ms" in metadata


def test_user_cv_service_with_real_user_files():
    """
    Integration test using real DocumentParser on user's actual CV files
    and MockProfileExtractor to verify complete service workflow without live LLM API keys.
    """
    pdf_path = Path("CV Nguyen Hai Nam.pdf")
    docx_path = Path("CV Nguyen Hai Nam.docx")

    if not (pdf_path.exists() and docx_path.exists()):
        pytest.skip("User CV files not present in project root.")

    mock_extractor = MockProfileExtractor()
    service = UserCVService(profile_extractor=mock_extractor)

    # Test PDF with real DocumentParser
    pdf_text = service.extract_raw_text(pdf_path)
    assert len(pdf_text) > 1000
    assert "NGUYEN HAI NAM" in pdf_text or "HANOI UNIVERSITY" in pdf_text.upper()

    profile_pdf, meta_pdf = service.process_cv_with_metadata(pdf_path)
    assert profile_pdf.full_name == "Le Van B"
    assert meta_pdf["raw_characters"] == len(pdf_text)
    assert meta_pdf["file_format"] == ".pdf"

    # Test DOCX with real DocumentParser
    docx_text = service.extract_raw_text(docx_path)
    assert len(docx_text) > 1000
    assert "NGUYEN HAI NAM" in docx_text or "HANOI UNIVERSITY" in docx_text.upper()

    profile_docx, meta_docx = service.process_cv_with_metadata(docx_path)
    assert profile_docx.full_name == "Le Van B"
    assert meta_docx["raw_characters"] == len(docx_text)
    assert meta_docx["file_format"] == ".docx"
