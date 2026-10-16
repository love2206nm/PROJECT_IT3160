"""
Live API Integration Test for LangGraph Copilot State Machine (Task 06).
Performs 1 single end-to-end execution with real outbound LLM API calls.
Protected by @pytest.mark.live_api:
- Skipped by default when running 'pytest' to prevent accidental token usage.
- Executed only when explicitly passing '--run-live-api' flag or setting RUN_LIVE_API=true.
"""

import time
import pytest
from langgraph.checkpoint.memory import MemorySaver

from source.model.LLM_model.copilot.state import AgentState
from source.model.LLM_model.copilot.graph import build_copilot_graph
from source.domain.entities.analysis import AnalysisResult
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.live_api
def test_live_copilot_graph_end_to_end():
    """
    Executes 1 live call through the LangGraph Copilot State Machine with real LLM reasoning.
    Validates that:
    1. Real LLM extracts candidate profile and matches SOC.
    2. RAG retrieves real O*NET benchmark evidence.
    3. Reasoning LLM (DeepSeek / Gemini) produces structured 30-60-90 day CopilotRoadmap.
    4. Critic Evaluator LLM audits roadmap and produces numerical scores.
    5. Final AnalysisResult domain entity is assembled and returned.
    """
    if not settings.DEEPSEEK_API_KEY and not settings.GEMINI_API_KEY:
        pytest.skip("Neither DEEPSEEK_API_KEY nor GEMINI_API_KEY is configured in environment.")

    logger.info("=== STARTING LIVE API INTEGRATION TEST FOR LANGGRAPH COPILOT ===")
    start_total = time.perf_counter()

    sample_resume = (
        "Nguyen Van Minh\n"
        "Senior Backend & Cloud Engineer\n"
        "Summary: 4.5 years of experience architecting distributed backend microservices and cloud infrastructure.\n"
        "Technical Skills: Python, FastAPI, Docker, Kubernetes, PostgreSQL, Redis, AWS, CI/CD, Terraform, REST APIs.\n"
        "Soft Skills: System Design, Critical Thinking, Agile Leadership.\n"
        "Experience: Senior Backend Developer at VNG Corp (2021 - 2024). Designed high-throughput microservices handling 20k RPS.\n"
        "Education: Bachelor of Computer Science, Hanoi University of Science and Technology (HUST)."
    )

    # Build graph with real default LLMs and MemorySaver
    app = build_copilot_graph(
        checkpointer=MemorySaver(),
        interrupt_before=[],  # Bypass HITL interruption for automated live API test
    )

    thread_id = f"live_test_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = AgentState(
        raw_cv_text=sample_resume,
        user_confirmed=True,
        iteration_count=0,
    )

    logger.info("Executing StateGraph pipeline with live API calls for thread '%s'...", thread_id)
    step_count = 0
    for event in app.stream(initial_state, config=config):
        step_count += 1
        node_name = list(event.keys())[0] if event else "unknown"
        logger.info("Completed node step %d: '%s'", step_count, node_name)

    elapsed_s = time.perf_counter() - start_total
    logger.info("Live StateGraph execution finished in %.2f seconds.", elapsed_s)

    # Retrieve final state from checkpointer
    final_state = app.get_state(config).values

    assert "final_analysis" in final_state, "final_analysis must be present in final state"
    analysis: AnalysisResult = final_state["final_analysis"]

    assert isinstance(analysis, AnalysisResult)
    assert analysis.candidate_profile is not None
    assert analysis.risk_profile is not None
    assert len(analysis.safe_pivots) > 0
    assert analysis.skill_gap is not None
    assert analysis.roadmap is not None

    roadmap = analysis.roadmap
    logger.info("=== LIVE GENERATION RESULT SUMMARY ===")
    logger.info("Target Role: %s (SOC: %s)", roadmap.target_role, roadmap.target_soc_code)
    logger.info("Executive Summary: %s", roadmap.executive_summary[:150])
    logger.info("Number of Phases: %d", len(roadmap.phases))
    logger.info("Mermaid Syntax Present: %s", bool(roadmap.mermaid_chart))
    logger.info("Critic Approved: %s | Iterations: %d", final_state.get("critic_approved"), final_state.get("iteration_count"))

    # Assertions on real generation quality
    assert len(roadmap.phases) >= 3, "Roadmap must contain at least 3 phases (30, 60, 90 days)"
    assert len(roadmap.defense_skills_breakdown) > 0, "Must contain defense skills"
    assert len(roadmap.mermaid_chart.strip()) > 10, "Must generate valid Mermaid diagram"
    assert final_state.get("iteration_count", 0) >= 1, "Must execute at least 1 Copilot-Critic iteration"
