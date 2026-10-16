"""
User CV Processing Service (Use Case Layer).
Orchestrates document parsing and deterministic profile extraction
for user-facing workflows (FastAPI endpoints, CLI, HITL confirmation).
"""

from pathlib import Path
import time
from typing import Optional, Tuple, Dict, Any

from source.domain.entities.candidate import CandidateProfile
from source.domain.interfaces.parser import IDocumentParser, IProfileExtractor
from source.domain.exceptions import CVExtractionError
from source.parsers.document_parser import DocumentParser
from source.parsers.deterministic_profile_parser import DeterministicProfileParser
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class UserCVService:
    """
    Application Service orchestrating CV ingestion and profile parsing.
    Acts as the entry point for candidate resume processing.
    Decoupled via IDocumentParser and IProfileExtractor interfaces.
    """

    def __init__(
        self,
        document_parser: Optional[IDocumentParser] = None,
        profile_extractor: Optional[IProfileExtractor] = None,
    ):
        """
        Initializes UserCVService with dependency injection.

        Args:
            document_parser: Implementation of IDocumentParser (defaults to DocumentParser)
            profile_extractor: Implementation of IProfileExtractor (defaults to DeterministicProfileParser)
        """
        self.document_parser = document_parser or DocumentParser()
        self.profile_extractor = profile_extractor or DeterministicProfileParser()
        logger.debug(
            "UserCVService initialized with parser: %s, extractor: %s",
            self.document_parser.__class__.__name__,
            self.profile_extractor.__class__.__name__,
        )

    def extract_raw_text(self, file_path: str | Path) -> str:
        """
        Step 1 of Ingestion: Extracts clean, layout-preserved text from a document.

        Args:
            file_path: Path to the CV document (.pdf, .docx, .txt)

        Returns:
            str: Cleaned text content preserving visual reading order.

        Raises:
            CVExtractionError: If file is missing, empty, or unreadable.
        """
        path = Path(file_path)
        logger.info("Extracting raw text from document: '%s'", path.name)
        start_time = time.perf_counter()

        try:
            raw_text = self.document_parser.extract_text(path)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Successfully extracted raw text from '%s' (%d chars, ~%d words) in %.2f ms",
                path.name,
                len(raw_text),
                len(raw_text.split()),
                elapsed_ms,
            )
            return raw_text
        except CVExtractionError:
            raise
        except Exception as e:
            logger.exception("Unexpected error while extracting raw text from '%s': %s", path.name, str(e))
            raise CVExtractionError(f"Failed to extract raw text from '{path.name}': {str(e)}") from e

    def extract_profile_from_text(self, raw_text: str) -> CandidateProfile:
        """
        Step 2 of Ingestion: Transforms raw CV text into structured CandidateProfile deterministically.

        Args:
            raw_text: Raw resume text (extracted from file or submitted directly by user)

        Returns:
            CandidateProfile: Validated domain entity.

        Raises:
            CVExtractionError: If text is too short or extraction fails.
        """
        if not raw_text or len(raw_text.strip()) < 30:
            logger.error("Raw resume text is empty or too short (< 30 characters).")
            raise CVExtractionError("Resume text is too short or empty for structured extraction.")

        logger.info("Extracting structured candidate profile deterministically (%d characters)...", len(raw_text))
        start_time = time.perf_counter()

        try:
            profile = self.profile_extractor.extract_profile(raw_text)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Structured profile extracted for '%s' (%s, %.1f yrs exp, %d skills) in %.2f ms",
                profile.full_name or "Unknown",
                profile.current_title,
                profile.years_of_experience,
                len(profile.get_all_skills),
                elapsed_ms,
            )
            return profile
        except CVExtractionError:
            raise
        except Exception as e:
            logger.exception("Error extracting candidate profile from text: %s", str(e))
            raise CVExtractionError(f"Profile extraction failed: {str(e)}") from e

    def process_cv_file(self, file_path: str | Path) -> CandidateProfile:
        """
        Full Pipeline: Reads a document file from disk and parses it directly into CandidateProfile.

        Args:
            file_path: Path to the target CV document

        Returns:
            CandidateProfile: Extracted, structured profile.
        """
        raw_text = self.extract_raw_text(file_path)
        return self.extract_profile_from_text(raw_text)

    def process_cv(self, file_path: str | Path) -> CandidateProfile:
        """Alias for process_cv_file conforming to test expectations."""
        return self.process_cv_file(file_path)

    def process_cv_with_metadata(self, file_path: str | Path) -> Tuple[CandidateProfile, Dict[str, Any]]:
        """
        Processes CV file and returns profile along with diagnostic metadata and latency telemetry.

        Returns:
            Tuple[CandidateProfile, Dict[str, Any]]: (profile, metadata)
        """
        path = Path(file_path)
        t0 = time.perf_counter()
        raw_text = self.extract_raw_text(path)
        t_text = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        profile = self.extract_profile_from_text(raw_text)
        t_profile = (time.perf_counter() - t1) * 1000

        metadata: Dict[str, Any] = {
            "file_name": path.name,
            "file_format": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size if path.exists() else 0,
            "raw_characters": len(raw_text),
            "estimated_tokens": max(1, len(raw_text) // 4),
            "text_extract_latency_ms": round(t_text, 2),
            "profile_extract_latency_ms": round(t_profile, 2),
            "total_latency_ms": round(t_text + t_profile, 2),
            "skills_extracted_count": len(profile.get_all_skills),
        }
        return profile, metadata
