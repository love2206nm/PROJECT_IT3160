"""
Career Analysis Result Retrieval Endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from source.api.schemas.analysis_schema import AnalysisDetailResponse
from source.api.dependencies import get_cv_analysis_service, CVAnalysisService

router = APIRouter()


@router.get(
    "/{session_id}",
    response_model=AnalysisDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Full Career Advisory Analysis Result",
    description="Fetches completed analysis, threat score, safe transitions, and 30-60-90 days roadmap for a given session.",
)
async def get_analysis_by_session(
    session_id: str,
    advisor: CVAnalysisService = Depends(get_cv_analysis_service),
) -> AnalysisDetailResponse:
    """Retrieves existing AnalysisResult from state memory."""
    analysis = advisor.get_analysis_result(session_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis found for session '{session_id}'. Please submit CV first.",
        )

    return AnalysisDetailResponse(
        session_id=session_id,
        status="COMPLETED",
        analysis=analysis,
    )
