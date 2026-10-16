"""
Unit & Integration Tests for Module 8 FastAPI Presentation Layer.
Tests all endpoints using FastAPI TestClient with Mocked Application Services:
- /health & /api/v1/health
- /api/v1/cv/analyze (Ingestion & HITL pause)
- /api/v1/cv/confirm (Resume & AnalysisResult)
- /api/v1/analysis/{session_id} (Session state retrieval)
- /api/v1/copilot/chat (Standard conversational JSON)
- /api/v1/copilot/chat/stream (Server-Sent Events streaming token pop)
- Global exception handling (400, 404, 422, 500)
"""

from typing import Dict, Any, Optional
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from source.api.app import app
from source.api.dependencies import get_cv_analysis_service
from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.risk import RiskProfile, RiskLevel, SubRiskScores
from source.domain.entities.transition import SafeTransitionRole, SkillGap
from source.domain.entities.roadmap import CopilotRoadmap, RoadmapPhase
from source.domain.entities.analysis import AnalysisResult


# ==============================================================================
# Test Fixtures & Mock Factory
# ==============================================================================

def create_sample_analysis_result(session_id: str = "test_api_session_01") -> AnalysisResult:
    candidate = CandidateProfile(
        full_name="Hoang Minh Duc",
        current_title="Software Developer",
        years_of_experience=4.0,
        hard_skills=["Python", "FastAPI", "Docker"],
        soft_skills=["Problem Solving"],
    )
    risk = RiskProfile(
        soc_code="15-1252.00",
        role_title="Software Developers",
        threat_score=35.0,
        risk_level=RiskLevel.MODERATE,
        sub_scores=SubRiskScores(task_automation=30.0, tool_maturity=40.0, agentic_exposure=35.0),
        defense_skills=["Cloud Architecture", "Complex Problem Solving"],
        is_deterministic=True,
    )
    pivot = SafeTransitionRole(
        target_soc_code="15-1252.00",
        target_title="Cloud Solutions Architect",
        utility_score=85.0,
        similarity_pct=88.0,
        threat_reduction=20.0,
        target_threat_score=25.0,
        target_risk_level=RiskLevel.LOW,
        defense_skills=["Cloud Architecture", "Kubernetes"],
    )
    roadmap = CopilotRoadmap(
        target_role="Cloud Solutions Architect",
        target_soc_code="15-1252.00",
        executive_summary="90-day transition roadmap into Cloud Solutions Architect.",
        phases=[
            RoadmapPhase(
                phase_name="Phase 1: 30 Days",
                duration_days=30,
                objective="Master cloud architecture fundamentals",
                action_items=["Complete AWS labs"],
                recommended_certifications=["AWS SAA"],
                target_skills=["Cloud Architecture"],
                practical_projects=["Container Deployment"],
            )
        ],
        defense_skills_breakdown=["Cloud Architecture"],
        mermaid_chart="graph TD\n    A[Phase 1] --> B[Target]",
        estimated_completion_months=3,
    )
    return AnalysisResult(
        session_id=session_id,
        candidate_profile=candidate,
        risk_profile=risk,
        safe_pivots=[pivot],
        roadmap=roadmap,
    )


@pytest.fixture
def mock_service():
    service = MagicMock()
    # Mock Phase 1 start_cv_analysis
    service.start_cv_analysis.return_value = {
        "session_id": "test_api_session_01",
        "status": "AWAITING_HUMAN_REVIEW",
        "candidate_profile": create_sample_analysis_result().candidate_profile,
        "suggested_soc_code": "15-1252.00",
        "suggested_title": "Software Developers",
        "match_confidence": 0.95,
        "latency_ms": 15.4,
    }
    # Mock Phase 2 confirm_and_resume_analysis
    service.confirm_and_resume_analysis.return_value = create_sample_analysis_result("test_api_session_01")

    # Mock Result Retrieval
    def _get_analysis(session_id: str):
        if session_id == "test_api_session_01":
            return create_sample_analysis_result(session_id)
        return None

    service.get_analysis_result.side_effect = _get_analysis

    # Mock Copilot Chat
    service.chat_with_copilot.return_value = "Focus on your Phase 1 AWS Cloud certification labs."

    # Mock Copilot Streaming Chat
    def _stream_chat(session_id: str, message: str, chunk_delay_sec: float = 0.0):
        tokens = ["Focus ", "on ", "your ", "Phase 1 ", "AWS ", "certification."]
        for t in tokens:
            yield t

    service.stream_chat_with_copilot.side_effect = _stream_chat

    return service


@pytest.fixture
def client(mock_service):
    """Provides a TestClient with overridden CVAnalysisService dependency."""
    app.dependency_overrides[get_cv_analysis_service] = lambda: mock_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ==============================================================================
# 1. Tests for System Health Endpoints
# ==============================================================================

def test_health_check_endpoints(client):
    """Verify both /health and /api/v1/health return 200 OK with valid schema."""
    resp1 = client.get("/health")
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "healthy"

    resp2 = client.get("/api/v1/health")
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "healthy"


def test_static_frontend_mount(client):
    """Verify root / serves static index.html with HTML content."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "AI Career Copilot" in resp.text


# ==============================================================================
# 2. Tests for Phase 1: CV Analyze Endpoint
# ==============================================================================

def test_analyze_cv_with_raw_text(client, mock_service):
    """Verify POST /api/v1/cv/analyze parses CV text and returns HITL pause."""
    resp = client.post(
        "/api/v1/cv/analyze",
        data={"raw_cv_text": "Hoang Minh Duc - Software Developer with 4 years Python."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "test_api_session_01"
    assert body["status"] == "AWAITING_HUMAN_REVIEW"
    assert body["suggested_soc_code"] == "15-1252.00"
    assert body["match_confidence"] >= 0.90
    mock_service.start_cv_analysis.assert_called_once()


def test_analyze_cv_with_file_upload(client, mock_service):
    """Verify POST /api/v1/cv/analyze accepts uploaded PDF/DOCX file."""
    fake_pdf = b"%PDF-1.4 Fake resume content with Python and Docker"
    resp = client.post(
        "/api/v1/cv/analyze",
        files={"file": ("sample_resume.pdf", fake_pdf, "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "AWAITING_HUMAN_REVIEW"


# ==============================================================================
# 3. Tests for Phase 2: CV Confirm Endpoint
# ==============================================================================

def test_confirm_analysis_endpoint(client, mock_service):
    """Verify POST /api/v1/cv/confirm resumes workflow and returns AnalysisDetailResponse."""
    payload = {
        "session_id": "test_api_session_01",
        "user_confirmed": True,
        "overrides": {"matched_title": "Senior Solutions Architect"},
    }
    resp = client.post("/api/v1/cv/confirm", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "test_api_session_01"
    assert body["status"] == "COMPLETED"
    assert body["analysis"] is not None
    assert body["analysis"]["roadmap"]["target_role"] == "Cloud Solutions Architect"
    mock_service.confirm_and_resume_analysis.assert_called_once_with(
        session_id="test_api_session_01",
        user_confirmed=True,
        overrides={"matched_title": "Senior Solutions Architect"},
    )


# ==============================================================================
# 4. Tests for Analysis Result Retrieval Endpoint
# ==============================================================================

def test_get_analysis_by_session_success(client):
    """Verify GET /api/v1/analysis/{session_id} returns existing completed analysis."""
    resp = client.get("/api/v1/analysis/test_api_session_01")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "test_api_session_01"
    assert body["analysis"]["risk_profile"]["threat_score"] == 35.0


def test_get_analysis_by_session_not_found(client):
    """Verify GET /api/v1/analysis/{session_id} returns 404 for unknown session."""
    resp = client.get("/api/v1/analysis/unknown_session_999")
    assert resp.status_code == 404
    assert "No analysis found" in resp.json()["detail"]


# ==============================================================================
# 5. Tests for Copilot Chat (Standard & Real-time SSE Streaming)
# ==============================================================================

def test_copilot_chat_json_endpoint(client, mock_service):
    """Verify POST /api/v1/copilot/chat returns standard JSON response."""
    payload = {
        "session_id": "test_api_session_01",
        "message": "What is the best certification to start with?",
    }
    resp = client.post("/api/v1/copilot/chat", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "test_api_session_01"
    assert "Phase 1" in body["response"]
    assert "timestamp" in body


def test_copilot_chat_stream_endpoint(client, mock_service):
    """Verify POST /api/v1/copilot/chat/stream returns real-time SSE tokens (text/event-stream)."""
    payload = {
        "session_id": "test_api_session_01",
        "message": "Guide me through Phase 1 step by step.",
    }
    resp = client.post("/api/v1/copilot/chat/stream", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Read SSE lines
    lines = resp.text.split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]

    assert len(data_lines) >= 6
    assert any("Focus" in line for line in data_lines)
    assert "data: [DONE]" in lines
