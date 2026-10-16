"""
Interactive AI Career Copilot Chat Endpoints (Standard JSON & Real-time SSE Streaming).
Supports both Conversational Follow-up and Direct CV Attachment Analysis.
"""

import json
from typing import Optional
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from source.api.schemas.chat_schema import CopilotChatRequest, CopilotChatResponse
from source.api.dependencies import get_cv_analysis_service, CVAnalysisService
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    response_model=CopilotChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Interactive Copilot Q&A (Standard JSON)",
    description="Answers user follow-up questions anchored in approved roadmap and dynamic O*NET Re-RAG evidence.",
)
async def chat_with_copilot(
    req: CopilotChatRequest,
    advisor: CVAnalysisService = Depends(get_cv_analysis_service),
) -> CopilotChatResponse:
    """Standard non-streaming Copilot conversational endpoint."""
    answer = advisor.chat_with_copilot(session_id=req.session_id, message=req.message)
    return CopilotChatResponse(
        session_id=req.session_id,
        response=answer,
    )


@router.post(
    "/chat/stream",
    status_code=status.HTTP_200_OK,
    summary="Interactive Copilot Q&A & CV Ingestion (Real-time SSE Stream)",
    description="Streams Copilot response token-by-token via Server-Sent Events. Supports both JSON message and Multipart CV file attachment.",
)
async def chat_with_copilot_stream(
    request: Request,
    advisor: CVAnalysisService = Depends(get_cv_analysis_service),
) -> StreamingResponse:
    """Streams continuous token chunks progressively using Server-Sent Events."""
    content_type = request.headers.get("content-type", "")

    session_id = ""
    message = ""
    file_bytes: Optional[bytes] = None
    filename: Optional[str] = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        session_id = str(form.get("session_id") or "")
        message = str(form.get("message") or "")
        file_obj = form.get("file")
        if file_obj and hasattr(file_obj, "read"):
            file_bytes = await file_obj.read()
            filename = getattr(file_obj, "filename", "resume.pdf")
    else:
        try:
            body = await request.json()
            session_id = str(body.get("session_id") or "")
            message = str(body.get("message") or "")
        except Exception:
            session_id = ""
            message = ""

    logger.info("Initiating SSE stream for session '%s' (message='%s', has_file=%s)", session_id, message[:40], file_bytes is not None)

    kwargs = {
        "session_id": session_id,
        "message": message,
    }
    if file_bytes is not None:
        kwargs["file_bytes"] = file_bytes
        kwargs["filename"] = filename

    def event_stream():
        try:
            for token in advisor.stream_chat_with_copilot(**kwargs):
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("Error during SSE stream: %s", e)
            error_payload = json.dumps({"error": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
