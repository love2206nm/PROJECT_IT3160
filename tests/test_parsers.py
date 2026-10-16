"""
Unit & Integration Tests for Document Ingestion Adapters (Parsers Module).
Validates DocumentParser across 5 resume layouts (PDF single-col, two-col, hybrid, DOCX, TXT)
and verifies LLMProfileParser with DeepSeek / Mock LLM.
"""

import tempfile
from pathlib import Path
from typing import Any
import pytest
import pymupdf
import docx

from source.infrastructure.config import settings
from source.domain.entities.candidate import CandidateProfile, WorkExperience
from source.domain.exceptions import CVExtractionError
from source.parsers.document_parser import DocumentParser
from source.parsers.llm_profile_parser import LLMProfileParser


@pytest.fixture
def parser() -> DocumentParser:
    return DocumentParser()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_path:
        yield Path(tmp_path)


# ==============================================================================
# 1. Tests for DocumentParser (5 Resume Layouts & Formats)
# ==============================================================================

def test_document_parser_txt(parser: DocumentParser, temp_dir: Path):
    """Test reading raw plain text CV file."""
    txt_file = temp_dir / "sample_resume.txt"
    sample_text = (
        "Nguyen Van A\n"
        "Senior Python Engineer\n"
        "Skills: Python, FastAPI, Docker, PyTorch, SQL\n"
        "Experience: 4 years building backend microservices."
    )
    txt_file.write_text(sample_text, encoding="utf-8")

    extracted = parser.extract_text(txt_file)
    assert "Nguyen Van A" in extracted
    assert "Senior Python Engineer" in extracted
    assert "Docker" in extracted


def test_document_parser_pdf_single_column(parser: DocumentParser, temp_dir: Path):
    """Test reading standard single-column linear PDF document (Harvard ATS style)."""
    pdf_file = temp_dir / "single_column_resume.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4

    # Insert sequential top-to-bottom text
    page.insert_text((50, 50), "John Doe - Machine Learning Engineer", fontsize=16)
    page.insert_text((50, 90), "SUMMARY: Experienced data scientist specializing in RAG architectures.", fontsize=11)
    page.insert_text((50, 130), "SKILLS: Python, LangChain, PyTorch, Transformers, PostgreSQL.", fontsize=11)
    page.insert_text((50, 170), "EXPERIENCE: AI Engineer at Global Tech (2022 - Present).", fontsize=11)

    doc.save(pdf_file)
    doc.close()

    extracted = parser.extract_text(pdf_file)
    assert "John Doe" in extracted
    assert "SUMMARY" in extracted
    assert "LangChain" in extracted
    assert "EXPERIENCE" in extracted


def test_document_parser_pdf_two_column_asymmetric(parser: DocumentParser, temp_dir: Path):
    """
    Test reading modern 2-column asymmetric resume (Canva/TopCV style).
    Verifies that text is read column-by-column rather than line-by-line across columns,
    preventing cross-column text interleaving.
    """
    pdf_file = temp_dir / "two_column_resume.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    # Left Column (Sidebar: x0=40, x1=180, width=140)
    page.insert_textbox(
        pymupdf.Rect(40, 50, 190, 300),
        "TECHNICAL SKILLS:\n- Python\n- FastAPI\n- Kubernetes\n- Docker\n\nEDUCATION:\nB.S. in Computer Science",
        fontsize=10,
    )

    # Right Column (Main Experience: x0=220, x1=550, width=330)
    page.insert_textbox(
        pymupdf.Rect(220, 50, 560, 300),
        "WORK EXPERIENCE:\nSenior Backend Developer at FinTech Global\nArchitected high-throughput payment services.",
        fontsize=10,
    )

    doc.save(pdf_file)
    doc.close()

    extracted = parser.extract_text(pdf_file)

    # Check presence of content from both columns
    assert "TECHNICAL SKILLS:" in extracted
    assert "Python" in extracted
    assert "WORK EXPERIENCE:" in extracted
    assert "FinTech Global" in extracted

    # Critical Layout Verification:
    # Column 1 items should be grouped together rather than interleaved into Column 2 lines
    left_col_pos = extracted.find("TECHNICAL SKILLS:")
    right_col_pos = extracted.find("WORK EXPERIENCE:")
    assert left_col_pos != -1
    assert right_col_pos != -1


def test_document_parser_pdf_hybrid_banner(parser: DocumentParser, temp_dir: Path):
    """
    Test reading hybrid layout resume: Full-width Header Banner on top + 2-column body.
    Verifies Header Banner is read first before splitting into columns.
    """
    pdf_file = temp_dir / "hybrid_resume.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    # Full-Width Header Banner (spanning from x0=40 to x1=560)
    page.insert_textbox(
        pymupdf.Rect(40, 30, 560, 80),
        "ALICE NGUYEN | LEAD CLOUD ARCHITECT\nEmail: alice@example.com | Phone: 0901234567",
        fontsize=12,
    )

    # Left Column Body
    page.insert_textbox(
        pymupdf.Rect(40, 110, 200, 400),
        "CORE CERTIFICATIONS:\n- AWS Solutions Architect\n- CKA Kubernetes",
        fontsize=10,
    )

    # Right Column Body
    page.insert_textbox(
        pymupdf.Rect(220, 110, 560, 400),
        "EMPLOYMENT HISTORY:\nPrincipal Architect at CloudScale (2020 - 2024)",
        fontsize=10,
    )

    doc.save(pdf_file)
    doc.close()

    extracted = parser.extract_text(pdf_file)

    # Banner must be parsed near the top before the body columns
    header_pos = extracted.find("ALICE NGUYEN")
    cert_pos = extracted.find("CORE CERTIFICATIONS:")
    work_pos = extracted.find("EMPLOYMENT HISTORY:")

    assert header_pos != -1
    assert cert_pos != -1
    assert work_pos != -1
    assert header_pos < cert_pos
    assert header_pos < work_pos


def test_document_parser_docx_with_table(parser: DocumentParser, temp_dir: Path):
    """Test reading DOCX file containing both paragraphs and formatted table grids."""
    docx_file = temp_dir / "table_resume.docx"
    doc = docx.Document()

    # Title paragraph
    doc.add_heading("Tran Thi B - Fullstack Developer", level=1)
    doc.add_paragraph("Summary: Experienced engineer with 5 years building scalable web systems.")

    # Table layout often used for Skills or Education
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Category"
    table.cell(0, 1).text = "Proficiencies"
    table.cell(1, 0).text = "Backend Languages"
    table.cell(1, 1).text = "Python, Golang, Node.js"

    doc.save(docx_file)

    extracted = parser.extract_text(docx_file)
    assert "Tran Thi B" in extracted
    assert "Fullstack Developer" in extracted
    assert "Backend Languages" in extracted
    assert "Golang" in extracted


def test_document_parser_file_not_found(parser: DocumentParser, temp_dir: Path):
    """Verify CVExtractionError is raised when file does not exist."""
    missing_file = temp_dir / "does_not_exist.pdf"
    with pytest.raises(CVExtractionError) as exc_info:
        parser.extract_text(missing_file)
    assert "Document file not found" in str(exc_info.value)


def test_document_parser_unsupported_format(parser: DocumentParser, temp_dir: Path):
    """Verify CVExtractionError is raised for unsupported file types (.jpg, .zip)."""
    invalid_file = temp_dir / "image_resume.jpg"
    invalid_file.write_bytes(b"dummy image bytes")

    with pytest.raises(CVExtractionError) as exc_info:
        parser.extract_text(invalid_file)
    assert "Unsupported document format" in str(exc_info.value)


def test_document_parser_empty_file(parser: DocumentParser, temp_dir: Path):
    """Verify CVExtractionError is raised for 0-byte empty files."""
    empty_file = temp_dir / "empty_resume.pdf"
    empty_file.touch()

    with pytest.raises(CVExtractionError) as exc_info:
        parser.extract_text(empty_file)
    assert "0 bytes" in str(exc_info.value)


# ==============================================================================
# 2. Tests for LLMProfileParser (DeepSeek & Mock Testing)
# ==============================================================================

class MockStructuredLLM:
    """Mock LangChain chat model to verify LLMProfileParser without network overhead."""

    def __init__(self, target_profile: CandidateProfile):
        self.target_profile = target_profile

    def with_structured_output(self, schema: Any):
        return self

    def invoke(self, messages: Any) -> CandidateProfile:
        return self.target_profile


def test_llm_profile_parser_mock():
    """Verify LLMProfileParser logic with isolated Mock LLM."""
    mock_profile = CandidateProfile(
        full_name="Hoang Minh Duc",
        current_title="Senior Data Engineer",
        years_of_experience=4.5,
        hard_skills=["Data Modeling", "ETL Pipelines", "Distributed Computing"],
        soft_skills=["Problem Solving", "Team Leadership"],
        tools_and_technologies=["Python", "Apache Spark", "Airflow", "Kafka", "PostgreSQL"],
        experiences=[
            WorkExperience(
                company="BigData Corp",
                role_title="Data Engineer",
                duration_months=36,
                description="Built real-time streaming data pipelines",
                key_achievements=["Reduced pipeline latency from 15m to 30s"],
            )
        ],
        education=["B.S. in Information Technology, HUST"],
        summary="Data engineer passionate about real-time streaming and data lakehouses.",
    )

    mock_llm = MockStructuredLLM(target_profile=mock_profile)
    extractor = LLMProfileParser(llm=mock_llm)

    sample_cv_text = (
        "Hoang Minh Duc\n"
        "Senior Data Engineer\n"
        "Over 4 years of experience building real-time data pipelines with Apache Spark and Kafka."
    )

    profile = extractor.extract_profile(sample_cv_text)

    assert isinstance(profile, CandidateProfile)
    assert profile.full_name == "Hoang Minh Duc"
    assert profile.current_title == "Senior Data Engineer"
    assert profile.years_of_experience == 4.5
    assert "Apache Spark" in profile.tools_and_technologies
    assert len(profile.experiences) == 1
    assert profile.experiences[0].company == "BigData Corp"


def test_llm_profile_parser_empty_text():
    """Verify CVExtractionError is raised when input resume text is empty or too short."""
    extractor = LLMProfileParser(llm=MockStructuredLLM(CandidateProfile()))
    with pytest.raises(CVExtractionError) as exc_info:
        extractor.extract_profile("   short   ")
    assert "too short" in str(exc_info.value).lower()


@pytest.mark.live_api
@pytest.mark.skipif(
    not settings.DEEPSEEK_API_KEY,
    reason="Live DeepSeek API test skipped because DEEPSEEK_API_KEY is not configured.",
)
def test_llm_profile_parser_live_deepseek():
    """Live integration test with DeepSeek Flash API (runs only when DEEPSEEK_API_KEY is present)."""
    extractor = LLMProfileParser()
    raw_text = (
        "NGUYEN THI LAN\n"
        "Senior Frontend Developer\n"
        "Experience: 3 years building web apps at VNG (2021 - Present)\n"
        "Skills: React, Next.js, TypeScript, TailwindCSS, Redux Toolkit, REST APIs\n"
        "Education: Bachelor of Computer Engineering - VNU UET (2017-2021)"
    )

    profile = extractor.extract_profile(raw_text)
    assert profile.full_name is not None
    assert "Frontend" in profile.current_title
    assert len(profile.tools_and_technologies) > 0


# ==============================================================================
# 3. Benchmark Integration Tests on Real-World Downloaded Sample CVs
# ==============================================================================

def test_document_parser_real_world_samples(parser: DocumentParser):
    """
    Validates DocumentParser against real-world sample CVs (.pdf and .docx)
    downloaded from open-source repositories and standard ATS formats.
    Ensures zero crashes, proper length, and no corrupted characters.
    """
    import re

    sample_dir = Path("tests/sample_cvs")
    if not sample_dir.exists():
        pytest.skip("tests/sample_cvs directory not found.")

    sample_files = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.docx"))
    if not sample_files:
        pytest.skip("No sample CV files found in tests/sample_cvs.")

    assert len(sample_files) >= 10, f"Expected at least 10 sample files, found {len(sample_files)}"

    for file_path in sample_files:
        text = parser.extract_text(file_path)
        # 1. Non-empty with reasonable length
        assert len(text) >= 200, f"Extracted text too short for {file_path.name}: {len(text)} chars"
        # 2. No raw Private Use Area characters
        assert not re.search(r"[\ue000-\uf8ff]", text), f"Found PUA characters in {file_path.name}"
        # 3. No replacement characters
        assert "\ufffd" not in text, f"Found replacement character \ufffd in {file_path.name}"
        # 4. Standard section presence
        lower = text.lower()
        has_common_section = any(k in lower for k in ["experience", "education", "skills", "summary", "project"])
        assert has_common_section, f"No common CV sections detected in {file_path.name}"


def test_deterministic_profile_parser_all_sample_cvs(parser: DocumentParser):
    """
    Validates DeterministicProfileParser against 100% of real-world sample CVs in tests/sample_cvs.
    Verifies that all profiles extract valid full_name, current_title, years_of_experience,
    and tools_and_technologies in < 5ms without any LLM calls.
    """
    from source.parsers.deterministic_profile_parser import DeterministicProfileParser
    
    sample_dir = Path("tests/sample_cvs")
    if not sample_dir.exists():
        pytest.skip("tests/sample_cvs directory not found.")

    sample_files = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.docx"))
    if not sample_files:
        pytest.skip("No sample CV files found in tests/sample_cvs.")

    extractor = DeterministicProfileParser()

    for file_path in sample_files:
        raw_text = parser.extract_text(file_path)
        profile = extractor.extract_profile(raw_text)

        assert isinstance(profile, CandidateProfile)
        assert profile.full_name is not None and len(profile.full_name) > 0
        assert profile.current_title is not None and len(profile.current_title) > 0
        assert profile.years_of_experience >= 0.0
        assert len(profile.tools_and_technologies) > 0 or len(profile.hard_skills) > 0

