"""
Document Ingestion Parser Adapter.
Implements IDocumentParser protocol:
- Layout-aware PDF block extraction using PyMuPDF (resolves 2-column & hybrid banner layouts).
- DOCX paragraph & table extraction using python-docx & docx2txt.
- Raw TXT reading with robust encoding fallbacks.
- Strict validation raising CVExtractionError.
"""

from pathlib import Path
from typing import List, Tuple
import pymupdf
import docx
import docx2txt

from source.domain.interfaces.parser import IDocumentParser
from source.domain.exceptions import CVExtractionError
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class DocumentParser(IDocumentParser):
    """
    Production-grade Document Parser Adapter.
    Preserves natural reading order across 5 major resume formats (1-column, 2-column, hybrid banner, tables, text).
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def extract_text(self, file_path: str | Path) -> str:
        """
        Extract clean, layout-preserved text from a document file.

        Args:
            file_path: Path to the CV document (.pdf, .docx, .txt)

        Returns:
            str: Cleaned, layout-preserved document text.

        Raises:
            CVExtractionError: When file does not exist, format is unsupported, or content is unreadable.
        """
        path = Path(file_path)

        # 1. Validate File Existence
        if not path.exists():
            logger.error("Document file not found: '%s'", path)
            raise CVExtractionError(f"Document file not found: '{path}'")

        if not path.is_file():
            logger.error("Path is not a regular file: '%s'", path)
            raise CVExtractionError(f"Path is not a valid file: '{path}'")

        # 2. Validate Supported Extension
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            logger.error("Unsupported file extension '%s' for '%s'", suffix, path.name)
            raise CVExtractionError(
                f"Unsupported document format: '{suffix}'. Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        # 3. Check for 0-byte Empty File
        if path.stat().st_size == 0:
            logger.error("Document file is empty (0 bytes): '%s'", path.name)
            raise CVExtractionError(f"Document file is empty (0 bytes): '{path.name}'")

        logger.info("Extracting text from document: '%s' (format: %s, size: %d bytes)...", path.name, suffix, path.stat().st_size)

        try:
            if suffix == ".pdf":
                raw_text = self._extract_pdf(path)
            elif suffix == ".docx":
                raw_text = self._extract_docx(path)
            elif suffix == ".txt":
                raw_text = self._extract_txt(path)
            else:
                raise CVExtractionError(f"Unhandled file extension: {suffix}")
        except CVExtractionError:
            raise
        except Exception as e:
            logger.exception("Unexpected error while parsing document '%s': %s", path.name, str(e))
            raise CVExtractionError(f"Failed to extract text from document '{path.name}': {str(e)}") from e

        # 4. Clean and Normalize Extracted Text
        clean_text = self._clean_extracted_text(raw_text)
        if not clean_text:
            logger.error("Document contains no readable text content: '%s'", path.name)
            raise CVExtractionError(f"Document contains no readable text content: '{path.name}'")

        logger.info("Successfully extracted %d characters from '%s'.", len(clean_text), path.name)
        return clean_text

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """
        Cleans and normalizes extracted text:
        - Replaces Wingdings/Symbol bullet points (\uf0b7, \uf0a7, etc.) with standard hyphen bullets.
        - Strips invisible Unicode Private Use Area characters (\ue000-\uf8ff).
        - Normalizes irregular whitespace, dashes, and replacement characters (\ufffd).
        """
        import re

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize bullets and special symbol characters to standard '-'
        bullet_pattern = re.compile(r"[\uf0b7\uf0a7\uf0d8\u2022\u25cf\u25cb\u25aa\u25ab\u25b8\u2023\ufffd\xad]")
        text = bullet_pattern.sub("-", text)

        # Remove remaining Private Use Area characters (common in exported PDFs with custom fonts)
        pua_pattern = re.compile(r"[\ue000-\uf8ff]")
        text = pua_pattern.sub("", text)

        # Convert common contact and section icons/emojis to meaningful text tags for LLM entity extraction
        emoji_map = {
            "\U0001f4cd": " [Location] ",   # 📍
            "\U0001f4e7": " [Email] ",      # 📧
            "\u2709": " [Email] ",          # ✉
            "\u2709\ufe0f": " [Email] ",
            "\u260e": " [Phone] ",          # ☎
            "\u260e\ufe0f": " [Phone] ",
            "\U0001f4de": " [Phone] ",      # 📞
            "\U0001f4f1": " [Mobile] ",     # 📱
            "\U0001f517": " [Link] ",       # 🔗
            "\U0001f310": " [Website] ",    # 🌐
            "\U0001f393": " [Education] ",  # 🎓
            "\U0001f4bc": " [Experience] ", # 💼
            "\u2b50": " [Award] ",          # ⭐
            "\U0001f3c6": " [Award] ",      # 🏆
        }
        for emo, repl in emoji_map.items():
            text = text.replace(emo, repl)

        # Normalize em-dash, en-dash, minus to standard hyphen
        text = re.sub(r"[\u2012\u2013\u2014\u2015]", "-", text)

        # Collapse excessive whitespace per line
        lines = []
        for line in text.split("\n"):
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(cleaned_line)

        # Collapse more than 2 consecutive blank lines
        result = "\n".join(lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()


    def _extract_pdf(self, path: Path) -> str:
        """
        Extracts text from PDF preserving multi-column and hybrid banner layout structure.
        Uses geometric bounding boxes (x0, y0, x1, y1) to detect parallel columns and
        read top header banners, left/right columns, and footers in natural reading order.
        """
        doc = pymupdf.open(path)
        if len(doc) == 0:
            doc.close()
            raise CVExtractionError(f"PDF document has 0 pages: '{path.name}'")

        page_texts: List[str] = []

        for page_idx, page in enumerate(doc):
            pw = page.rect.width
            # Extract text blocks: tuple (x0, y0, x1, y1, text, block_no, block_type)
            # block_type 0 = text, 1 = image
            raw_blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]

            if not raw_blocks:
                continue

            # Calculate horizontal centers for each block
            # Identify approximate page midline (default to 0.5 * pw or median of centers)
            centers = [(b[0] + b[2]) / 2.0 for b in raw_blocks]
            mid_x = pw / 2.0

            # Detect if there are parallel blocks across the midline (true multi-column zone)
            parallel_pairs = []
            for i in range(len(raw_blocks)):
                b_i = raw_blocks[i]
                c_i = (b_i[0] + b_i[2]) / 2.0
                for j in range(i + 1, len(raw_blocks)):
                    b_j = raw_blocks[j]
                    c_j = (b_j[0] + b_j[2]) / 2.0

                    # Check if one is on the left and the other on the right
                    is_cross_mid = (c_i < mid_x and c_j >= mid_x) or (c_j < mid_x and c_i >= mid_x)
                    # Check vertical overlap
                    y_overlap = max(b_i[1], b_j[1]) < min(b_i[3], b_j[3])

                    if is_cross_mid and y_overlap:
                        parallel_pairs.append((b_i, b_j))

            # Case A: If no parallel cross-column blocks exist, treat as 1-column linear page
            if not parallel_pairs:
                sorted_blocks = sorted(raw_blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
                page_text = "\n\n".join(b[4].strip() for b in sorted_blocks if b[4].strip())
                page_texts.append(page_text)
                continue

            # Case B: Multi-column layout with possible Top/Bottom full-width banners
            # Find the vertical span of the 2-column body
            col_zone_top = min(min(b_i[1], b_j[1]) for b_i, b_j in parallel_pairs)
            col_zone_bottom = max(max(b_i[3], b_j[3]) for b_i, b_j in parallel_pairs)

            top_banner = []
            bottom_banner = []
            col_left = []
            col_right = []

            for b in raw_blocks:
                c_x = (b[0] + b[2]) / 2.0
                # Top Banner: blocks positioned strictly above the parallel column zone
                if b[3] <= col_zone_top + 5.0:
                    top_banner.append(b)
                # Bottom Banner: blocks positioned strictly below the parallel column zone
                elif b[1] >= col_zone_bottom - 5.0:
                    bottom_banner.append(b)
                # Middle Column Zone: partition into Left vs Right
                elif c_x < mid_x:
                    col_left.append(b)
                else:
                    col_right.append(b)

            # Sort each section top-to-bottom
            top_sorted = sorted(top_banner, key=lambda b: (round(b[1], 1), round(b[0], 1)))
            bottom_sorted = sorted(bottom_banner, key=lambda b: (round(b[1], 1), round(b[0], 1)))
            col_left_sorted = sorted(col_left, key=lambda b: b[1])
            col_right_sorted = sorted(col_right, key=lambda b: b[1])

            # Assemble natural reading sequence:
            # Top Header Banner -> Left Column (Skills/Sidebar) -> Right Column (Experience/Main) -> Bottom Footer
            ordered_blocks = []
            ordered_blocks.extend(top_sorted)
            ordered_blocks.extend(col_left_sorted)
            ordered_blocks.extend(col_right_sorted)
            ordered_blocks.extend(bottom_sorted)

            page_text = "\n\n".join(b[4].strip() for b in ordered_blocks if b[4].strip())
            if page_text:
                page_texts.append(page_text)

        doc.close()
        return "\n\n--- PAGE BREAK ---\n\n".join(page_texts)

    def _extract_docx(self, path: Path) -> str:
        """
        Extracts text from DOCX files, reading both standard paragraphs and table grids.
        """
        text_parts: List[str] = []

        try:
            doc = docx.Document(path)
            # 1. Extract paragraphs
            for p in doc.paragraphs:
                p_text = p.text.strip()
                if p_text:
                    text_parts.append(p_text)

            # 2. Extract tables (grid layouts often used for resume sections)
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    # Deduplicate adjacent cells if merged
                    unique_cells = []
                    for c in row_cells:
                        if not unique_cells or c != unique_cells[-1]:
                            unique_cells.append(c)
                    if unique_cells:
                        text_parts.append(" | ".join(unique_cells))

            if text_parts:
                return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning("python-docx extraction failed for '%s' (%s), trying docx2txt fallback...", path.name, str(e))

        # Fallback to docx2txt
        fallback_text = docx2txt.process(str(path))
        if fallback_text and fallback_text.strip():
            return fallback_text.strip()

        raise CVExtractionError(f"No readable text extracted from DOCX document: '{path.name}'")

    def _extract_txt(self, path: Path) -> str:
        """
        Reads raw text file with UTF-8 encoding, falling back to latin-1.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                return f.read()
