"""
Unit & Integration Tests for LangGraph Copilot State Machine (Task 06).
100% offline, zero-network, zero token cost testing using Mock LLM fixtures.
Validates:
- State schema & domain entities
- 3-tier routing & threshold enforcement (0.90)
- Human-in-the-Loop interruption & state resumption with MemorySaver
- Self-correction Critic reflection loop (reject & retry cycle)
- Dual-mode conversational memory & Dynamic Re-RAG isolation
- End-to-end unified AnalysisResult generation
"""

from typing import Any, List, Dict
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from source.model.LLM_model.copilot.state import AgentState
from source.model.LLM_model.copilot.graph import (
    build_copilot_graph,
    route_entrypoint,
    route_after_matcher,
    route_after_critic,
)
from source.model.LLM_model.copilot.nodes import (
    cv_ingest_node,
    deterministic_lookup,
    ml_scoring_node,
    cluster_pivot_node,
    assemble_result_node,
    interactive_copilot_node,
)
from source.model.LLM_model.copilot.prompts import (
    CriticEvaluation,
    CriticScores,
)
from source.domain.entities.candidate import CandidateProfile, WorkExperience
from source.domain.entities.risk import RiskProfile, RiskLevel, SubRiskScores
from source.domain.entities.transition import SafeTransitionRole, SkillGap
from source.domain.entities.roadmap import CopilotRoadmap, RoadmapPhase
from source.domain.entities.analysis import AnalysisResult


# ==============================================================================
# Mock LLM Fixtures & Helper Classes
# ==============================================================================

class MockStructuredLLM:
    """Mock LangChain chat model to simulate structured LLM generation with zero network calls."""

    def __init__(self, responses: List[Any]):
        self.responses = list(responses)
        self.call_count = 0

    def with_structured_output(self, schema: Any):
        return self

    def invoke(self, messages: Any) -> Any:
        self.call_count += 1
        if self.responses:
            res = self.responses.pop(0)
            if callable(res):
                return res(messages)
            return res
        return {}


def create_sample_candidate() -> CandidateProfile:
    return CandidateProfile(
        full_name="Hoang Minh Duc",
        current_title="Software Developer",
        years_of_experience=4.0,
        hard_skills=["Python", "FastAPI", "Docker", "PostgreSQL", "REST APIs"],
        soft_skills=["Critical Thinking", "Problem Solving"],
        tools_and_technologies=["Git", "Linux", "Docker", "FastAPI"],
        experiences=[
            WorkExperience(
                company="Tech Corp",
                role_title="Backend Developer",
                duration_months=36,
                description="Built distributed microservices",
            )
        ],
    )


def create_sample_roadmap(target_role: str = "Cloud & AI Solutions Architect") -> CopilotRoadmap:
    phase1 = RoadmapPhase(
        phase_name="Phase 1: 30 Days - Foundation",
        duration_days=30,
        objective="Master cloud architecture fundamentals and IaC",
        action_items=["Complete Terraform Associate labs", "Build Docker Compose microservice"],
        recommended_certifications=["AWS Certified Solutions Architect - Associate"],
        target_skills=["Cloud Architecture", "Terraform"],
        practical_projects=["Automated Container Deployment"],
    )
    phase2 = RoadmapPhase(
        phase_name="Phase 2: 60 Days - Deepening",
        duration_days=30,
        objective="Deploy scalable Kubernetes workloads and CI/CD pipelines",
        action_items=["Configure Helm charts", "Deploy Prometheus/Grafana stack"],
        recommended_certifications=["Certified Kubernetes Administrator (CKA)"],
        target_skills=["Kubernetes", "Observability"],
        practical_projects=["Production-grade K8s Cluster"],
    )
    phase3 = RoadmapPhase(
        phase_name="Phase 3: 90 Days - Mastery",
        duration_days=30,
        objective="Deliver end-to-end cloud AI capstone and career pivot portfolio",
        action_items=["Publish capstone repo", "Present technical demo"],
        recommended_certifications=["AWS Certified Solutions Architect - Professional"],
        target_skills=["Enterprise Architecture", "AI Orchestration"],
        practical_projects=["End-to-End Enterprise AI Pipeline"],
    )

    return CopilotRoadmap(
        target_role=target_role,
        target_soc_code="15-1252.00",
        executive_summary=f"Tailored 90-day transition roadmap into {target_role}.",
        phases=[phase1, phase2, phase3],
        defense_skills_breakdown=["Cloud Architecture", "Systems Thinking", "Kubernetes"],
        mermaid_chart="graph TD\n    A[Phase 1] --> B[Phase 2] --> C[Phase 3]",
        estimated_completion_months=3,
    )


# ==============================================================================
# 1. Tests for State Schema & Initial Routing
# ==============================================================================

def test_state_schema_keys():
    """Verify AgentState schema keys match domain requirements."""
    state = AgentState(
        raw_cv_text="sample cv",
        iteration_count=0,
        user_confirmed=False,
    )
    assert state["raw_cv_text"] == "sample cv"
    assert state["iteration_count"] == 0
    assert state["user_confirmed"] is False


def test_route_entrypoint_fresh_vs_interactive():
    """Verify entrypoint routes new sessions to ingest and ongoing chat to interactive copilot."""
    fresh_state = AgentState(raw_cv_text="Hello CV", iteration_count=0)
    assert route_entrypoint(fresh_state) == "cv_ingest_node"

    resumed_state = AgentState(
        final_analysis=AnalysisResult(
            candidate_profile=create_sample_candidate(),
            risk_profile=RiskProfile(role_title="Software Developer", threat_score=35.0, risk_level=RiskLevel.LOW),
        ),
        messages=[HumanMessage(content="What cert should I take first?")],
    )
    assert route_entrypoint(resumed_state) == "interactive_copilot_node"


# ==============================================================================
# 2. Tests for 3-Tier Matcher & Router Thresholds
# ==============================================================================

def test_route_after_matcher_high_confidence():
    """Verify confidence >= 0.90 routes to deterministic_lookup."""
    state = AgentState(
        matched_soc_code="15-1252",
        match_confidence=0.95,
        user_confirmed=False,
    )
    assert route_after_matcher(state) == "deterministic_lookup"


def test_route_after_matcher_user_confirmed_override():
    """Verify user confirmation routes to deterministic_lookup even with lower initial confidence."""
    state = AgentState(
        matched_soc_code="15-1252",
        match_confidence=0.75,
        user_confirmed=True,
    )
    assert route_after_matcher(state) == "deterministic_lookup"


def test_route_after_matcher_low_confidence_fallback():
    """Verify confidence < 0.90 without confirmation routes to ml_scoring_node."""
    state = AgentState(
        matched_soc_code="15-1252",
        match_confidence=0.82,  # e.g. Dairy Scientist style false positive score
        user_confirmed=False,
    )
    assert route_after_matcher(state) == "ml_scoring_node"


# ==============================================================================
# 3. Tests for Critic Routing & Ceiling Enforcement
# ==============================================================================

def test_route_after_critic_approved():
    """Verify critic approval immediately exits to assemble_result_node."""
    state = AgentState(critic_approved=True, iteration_count=1)
    assert route_after_critic(state) == "assemble_result_node"


def test_route_after_critic_rejected_retry():
    """Verify critic rejection on iteration 1 loops back to copilot_agent_node."""
    state = AgentState(critic_approved=False, iteration_count=1)
    assert route_after_critic(state) == "copilot_agent_node"


def test_route_after_critic_max_iterations_ceiling():
    """Verify critic rejection on iteration >= 2 forces exit to assemble_result_node."""
    state = AgentState(critic_approved=False, iteration_count=2)
    assert route_after_critic(state) == "assemble_result_node"


# ==============================================================================
# 4. Tests for Specific Functional Nodes
# ==============================================================================

def test_cv_ingest_node_with_text():
    """Verify cv_ingest_node cleans text and initializes iteration_count."""
    state = AgentState(raw_cv_text="  Sample Cleaned Resume Text  ")
    out = cv_ingest_node(state)
    assert out["raw_cv_text"] == "Sample Cleaned Resume Text"
    assert out["iteration_count"] == 0


def test_deterministic_lookup_node():
    """Verify deterministic lookup returns authoritative RiskProfile with is_deterministic=True."""
    state = AgentState(matched_soc_code="15-1252", matched_title="Software Developers")
    out = deterministic_lookup(state)
    assert "risk_profile" in out
    risk: RiskProfile = out["risk_profile"]
    assert risk.is_deterministic is True
    assert risk.confidence == 1.0
    assert risk.soc_code == "15-1252"


def test_ml_scoring_node():
    """Verify ml_scoring_node returns RiskProfile with is_deterministic=False."""
    candidate = create_sample_candidate()
    state = AgentState(candidate_profile=candidate)
    out = ml_scoring_node(state)
    assert "risk_profile" in out
    risk: RiskProfile = out["risk_profile"]
    assert risk.is_deterministic is False
    assert 0.0 <= risk.threat_score <= 100.0


def test_cluster_pivot_node():
    """Verify cluster_pivot_node generates Top 3 safe pivots and builds SkillGap."""
    state = AgentState(
        matched_soc_code="15-1252",
        candidate_profile=create_sample_candidate(),
        risk_profile=RiskProfile(role_title="Software Developers", threat_score=50.0, risk_level=RiskLevel.MODERATE),
    )
    out = cluster_pivot_node(state)
    assert "safe_pivot_roles" in out
    assert "skill_gap" in out
    assert len(out["safe_pivot_roles"]) > 0
    assert isinstance(out["skill_gap"], SkillGap)
    assert out["skill_gap"].target_role_title != ""


# ==============================================================================
# 5. Tests for Human-in-the-Loop (HITL) Interruption & Resumption
# ==============================================================================

def test_hitl_interruption_and_resumption():
    """
    Verify graph pauses at human_review_node checkpoint,
    allows state inspection/modification, and resumes to completion.
    """
    candidate = create_sample_candidate()
    mock_roadmap = create_sample_roadmap()
    mock_critic = CriticEvaluation(
        approved=True,
        feedback="Excellent roadmap.",
        scores=CriticScores(feasibility=9.0, grounding=9.0, hallucination_free=9.5),
    )

    mock_copilot_llm = MockStructuredLLM([mock_roadmap])
    mock_critic_llm = MockStructuredLLM([mock_critic])

    memory = MemorySaver()
    app = build_copilot_graph(
        copilot_llm=mock_copilot_llm,
        critic_llm=mock_critic_llm,
        checkpointer=memory,
        interrupt_before=["human_review_node"],
    )

    thread_id = "test_hitl_thread_01"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = AgentState(
        raw_cv_text="Hoang Minh Duc. Senior Backend Engineer with Python, FastAPI, Docker.",
        candidate_profile=candidate,
        matched_soc_code="15-1252",
        matched_title="Software Developers",
        match_confidence=0.95,
        iteration_count=0,
    )

    # 1. Run until HITL interrupt
    for _ in app.stream(initial_state, config=config):
        pass

    state_snapshot = app.get_state(config)
    assert "human_review_node" in state_snapshot.next, "Graph must pause immediately before human_review_node"
    assert state_snapshot.values.get("matched_soc_code") == "15-1252"

    # 2. Simulate User Confirmation / Override in HITL
    app.update_state(config, {"user_confirmed": True, "matched_title": "Senior Cloud Engineer"})

    # 3. Resume Graph to completion
    final_output = None
    for event in app.stream(None, config=config):
        final_output = event

    final_snapshot = app.get_state(config)
    assert len(final_snapshot.next) == 0, "Graph should finish execution"
    assert "final_analysis" in final_snapshot.values
    analysis: AnalysisResult = final_snapshot.values["final_analysis"]
    assert analysis.roadmap is not None
    assert analysis.roadmap.target_role == mock_roadmap.target_role


# ==============================================================================
# 6. Tests for Critic Self-Correction Reflection Loop
# ==============================================================================

def test_critic_self_correction_cycle():
    """
    Simulates:
    - Iteration 1: Copilot generates Draft 1 -> Critic rejects with actionable feedback.
    - Iteration 2: Copilot incorporates feedback, generates Draft 2 -> Critic approves -> completes.
    """
    candidate = create_sample_candidate()
    draft1_roadmap = create_sample_roadmap(target_role="Draft 1 Role")
    draft2_roadmap = create_sample_roadmap(target_role="Draft 2 Refined Role")

    critic_rejection = CriticEvaluation(
        approved=False,
        feedback="Phase 1 timeline too compressed. Re-orient toward foundational containerization.",
        scores=CriticScores(feasibility=5.0, grounding=8.0, hallucination_free=8.5),
    )
    critic_approval = CriticEvaluation(
        approved=True,
        feedback="Phase 1 restructured realistically. Approved.",
        scores=CriticScores(feasibility=9.0, grounding=9.0, hallucination_free=9.5),
    )

    # Sequence of responses: Copilot(draft1) -> Critic(reject) -> Copilot(draft2) -> Critic(approve)
    mock_copilot_llm = MockStructuredLLM([draft1_roadmap, draft2_roadmap])
    mock_critic_llm = MockStructuredLLM([critic_rejection, critic_approval])

    memory = MemorySaver()
    app = build_copilot_graph(
        copilot_llm=mock_copilot_llm,
        critic_llm=mock_critic_llm,
        checkpointer=memory,
        interrupt_before=[],  # No HITL pause for automated test
    )

    config = {"configurable": {"thread_id": "test_critic_loop_02"}}
    initial_state = AgentState(
        raw_cv_text="Software Developer CV with Python, Docker, SQL.",
        candidate_profile=candidate,
        user_confirmed=True,
        matched_soc_code="15-1252",
        matched_title="Software Developers",
        match_confidence=1.0,
        iteration_count=0,
    )

    for _ in app.stream(initial_state, config=config):
        pass

    final_snapshot = app.get_state(config)
    assert final_snapshot.values.get("iteration_count") == 2, "Must execute exactly 2 iterations"
    assert final_snapshot.values.get("critic_approved") is True
    assert final_snapshot.values.get("copilot_roadmap").target_role == "Draft 2 Refined Role"
    assert "final_analysis" in final_snapshot.values


# ==============================================================================
# 7. Tests for End-to-End Workflow & Multi-Turn Interactive Copilot
# ==============================================================================

def test_end_to_end_mocked_graph():
    """Verify complete end-to-end execution packaging a valid AnalysisResult."""
    candidate = create_sample_candidate()
    mock_roadmap = create_sample_roadmap()
    mock_critic = CriticEvaluation(
        approved=True,
        feedback="Approved.",
        scores=CriticScores(feasibility=8.5, grounding=8.5, hallucination_free=9.0),
    )

    mock_copilot_llm = MockStructuredLLM([mock_roadmap])
    mock_critic_llm = MockStructuredLLM([mock_critic])

    app = build_copilot_graph(
        copilot_llm=mock_copilot_llm,
        critic_llm=mock_critic_llm,
        checkpointer=MemorySaver(),
        interrupt_before=[],
    )

    config = {"configurable": {"thread_id": "test_e2e_03"}}
    state = AgentState(
        raw_cv_text="Full resume content for Hoang Minh Duc, Python Developer.",
        candidate_profile=candidate,
        user_confirmed=True,
        matched_soc_code="15-1252",
        matched_title="Software Developers",
        match_confidence=0.95,
        iteration_count=0,
    )

    for _ in app.stream(state, config=config):
        pass

    final_snapshot = app.get_state(config)
    analysis: AnalysisResult = final_snapshot.values.get("final_analysis")

    assert analysis is not None
    assert analysis.candidate_profile.full_name == "Hoang Minh Duc"
    assert analysis.risk_profile.threat_score >= 0.0
    assert len(analysis.safe_pivots) > 0
    assert analysis.skill_gap is not None
    assert analysis.roadmap is not None
    assert "graph TD" in analysis.roadmap.mermaid_chart


def test_multi_turn_interactive_copilot_preserves_ground_truth():
    """
    Verify follow-up user questions route to interactive_copilot_node,
    leave original AnalysisResult untouched, and append conversational AIMessage.
    """
    candidate = create_sample_candidate()
    roadmap = create_sample_roadmap()
    analysis = AnalysisResult(
        candidate_profile=candidate,
        risk_profile=RiskProfile(role_title="Software Developer", threat_score=40.0, risk_level=RiskLevel.MODERATE),
        roadmap=roadmap,
    )

    # Mock response for conversational query
    class MockChatLLM:
        def invoke(self, messages):
            return AIMessage(content="Start with Phase 1: AWS Certified Solutions Architect [O*NET: 15-1252.00 - Certifications].")

    app = build_copilot_graph(
        copilot_llm=MockChatLLM(),
        checkpointer=MemorySaver(),
        interrupt_before=[],
    )

    config = {"configurable": {"thread_id": "test_multiturn_04"}}
    conversation_state = AgentState(
        final_analysis=analysis,
        skill_gap=SkillGap(target_role_title="Cloud Architect", target_soc_code="15-1252"),
        messages=[HumanMessage(content="Which certification should I tackle first in month 1?")],
    )

    for _ in app.stream(conversation_state, config=config):
        pass

    final_snapshot = app.get_state(config)
    # Ground truth analysis must remain pristine
    assert final_snapshot.values.get("final_analysis") == analysis
    # AIMessage appended
    messages = final_snapshot.values.get("messages", [])
    assert len(messages) >= 2
    assert "AWS Certified Solutions Architect" in messages[-1].content
    assert "[O*NET:" in messages[-1].content
