"""
Document Parser & Profile Extractor Domain Protocols (Ports).
Defines abstract contracts for CV extraction and LLM profile parsing.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable
from source.domain.entities.candidate import CandidateProfile


@runtime_checkable
class IDocumentParser(Protocol):
    """Protocol for extracting clean, layout-preserved text from PDF and DOCX files."""

    def extract_text(self, file_path: Path) -> str:
        """
        Extract clean text from a document file preserving visual reading order.

        Args:
            file_path: Absolute or relative Path to document (PDF/DOCX)

        Returns:
            str: Normalized raw text content

        Raises:
            CVExtractionError: If file is missing, corrupted, or unsupported
        """
        ...


@runtime_checkable
class IProfileExtractor(Protocol):
    """Protocol for transforming raw resume text into structured CandidateProfile entity via Fast LLM."""

    def extract_profile(self, raw_text: str) -> CandidateProfile:
        """
        Parse raw resume text into standardized domain CandidateProfile.

        Args:
            raw_text: Text extracted from candidate's resume

        Returns:
            CandidateProfile: Validated domain entity

        Raises:
            CVExtractionError: If LLM fails to extract valid profile structure
        """
        ...
