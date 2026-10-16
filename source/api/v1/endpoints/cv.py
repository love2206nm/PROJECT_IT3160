"""
Candidate CV Ingestion and Human-in-the-Loop Confirmation Endpoints.
"""

from pathlib import Path
import tempfile
from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, Form, status

from source.api.schemas.cv_schema import AnalyzeCVResponse, ConfirmAnalysisRequest
from source.api.schemas.analysis_schema import AnalysisDetailResponse
from source.api.dependencies import get_cv_analysis_service, CVAnalysisService
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeCVResponse,
    status_code=status.HTTP_200_OK,
    summary="Phase 1: Ingest CV & Pause at Human-in-the-Loop Checkpoint",
    description="Accepts resume file (.pdf, .docx, .txt) or raw text. Runs extraction and 3-Tier SOC matching, then pauses for human review in <3s.",
)
async def analyze_cv(
    file: Optional[UploadFile] = File(None, description="Resume file (.pdf, .docx, .txt)"),
    raw_cv_text: Optional[str] = Form(None, description="Direct resume text"),
    session_id: Optional[str] = Form(None, description="Optional custom session thread ID"),
    advisor: CVAnalysisService = Depends(get_cv_analysis_service),
) -> AnalyzeCVResponse:
    """Ingests CV document or raw text and pauses right before human_review_node."""
    temp_path: Optional[Path] = None

    try:
        if file and file.filename:
            suffix = Path(file.filename).suffix or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_path = Path(tmp.name)
            logger.info("Saved uploaded CV to temp file: %s (%d bytes)", temp_path, len(content))

        result = advisor.start_cv_analysis(
            file_path=temp_path,
            raw_cv_text=raw_cv_text,
            session_id=session_id,
        )

        return AnalyzeCVResponse(
            session_id=result["session_id"],
            status=result["status"],
            candidate_profile=result["candidate_profile"],
            suggested_soc_code=result["suggested_soc_code"],
            suggested_title=result["suggested_title"],
            match_confidence=result["match_confidence"],
            latency_ms=result.get("latency_ms", 0.0),
        )
    finally:
        # Clean up temporary uploaded file safely
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.debug("Could not unlink temp file %s: %s", temp_path, e)


@router.post(
    "/confirm",
    response_model=AnalysisDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Phase 2: Confirm Occupation & Execute Deep Career Analysis",
    description="Resumes execution after user validates or overrides occupation. Generates RiskProfile, Safe Pivots, RAG evidence, and Copilot Roadmap.",
)
async def confirm_and_resume_analysis(
    req: ConfirmAnalysisRequest,
    advisor: CVAnalysisService = Depends(get_cv_analysis_service),
) -> AnalysisDetailResponse:
    """Applies human review confirmation and runs the LangGraph Copilot workflow to completion."""
    analysis = advisor.confirm_and_resume_analysis(
        session_id=req.session_id,
        user_confirmed=req.user_confirmed,
        overrides=req.overrides,
    )
    return AnalysisDetailResponse(
        session_id=req.session_id,
        status="COMPLETED",
        analysis=analysis,
    )
