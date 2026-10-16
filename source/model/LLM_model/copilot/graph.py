"""
LangGraph Multi-Agent State Machine Assembly for AI Career Copilot.
Orchestrates 10 core nodes with MemorySaver checkpointing, Human-in-the-Loop interruption,
dynamic router branching, and self-correcting Critic reflection loop.
"""

from typing import Optional, List, Any, Dict, TYPE_CHECKING
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

if TYPE_CHECKING:
    from source.services.user.career_service import UserCareerService

from source.model.LLM_model.copilot.state import AgentState
from source.model.LLM_model.copilot.nodes import (
    cv_ingest_node,
    extract_profile_node,
    human_review_node,
    deterministic_lookup,
    ml_scoring_node,
    cluster_pivot_node,
    rag_grounding_node,
    copilot_agent_node,
    critic_evaluator_node,
    assemble_result_node,
    interactive_copilot_node,
)
from source.domain.interfaces.onet_repository import IOnetRepository
from source.domain.interfaces.rag import IEvidenceRetriever
from source.parsers.document_parser import DocumentParser
from source.parsers.llm_profile_parser import LLMProfileParser
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Conditional Edge Functions
# ==============================================================================

def route_entrypoint(state: AgentState) -> str:
    """
    Decides entrypoint:
    - If final_analysis already exists and user sent a new message -> 'interactive_copilot_node'
    - Otherwise -> 'cv_ingest_node'
    """
    if state.get("final_analysis") is not None and state.get("messages"):
        logger.info("[Routing] Resuming active session for multi-turn conversational Q&A.")
        return "interactive_copilot_node"
    logger.info("[Routing] Starting fresh career advisory ingestion pipeline.")
    return "cv_ingest_node"


def route_after_matcher(state: AgentState) -> str:
    """
    Conditional routing after Human Review:
    - If match_confidence >= ROUTER_CONFIDENCE_THRESHOLD (0.90) or user confirmed valid SOC -> 'deterministic_lookup'
    - Otherwise (custom/hybrid role) -> 'ml_scoring_node'
    """
    confidence = state.get("match_confidence", 0.0)
    user_confirmed = state.get("user_confirmed", False)
    soc_code = state.get("matched_soc_code")

    is_high_conf = confidence >= settings.ROUTER_CONFIDENCE_THRESHOLD
    if (is_high_conf or user_confirmed) and soc_code:
        logger.info(
            "[Routing] Route -> 'deterministic_lookup' (SOC: %s, Conf: %.3f, Confirmed: %s)",
            soc_code,
            confidence,
            user_confirmed,
        )
        return "deterministic_lookup"

    logger.info(
        "[Routing] Route -> 'ml_scoring_node' (Confidence %.3f < %.2f)",
        confidence,
        settings.ROUTER_CONFIDENCE_THRESHOLD,
    )
    return "ml_scoring_node"


def route_after_critic(state: AgentState) -> str:
    """
    Conditional routing for Critic reflection loop:
    - If critic_approved is True OR iteration_count >= MAX_CRITIC_ITERATIONS (2) -> 'assemble_result_node'
    - Otherwise -> 'copilot_agent_node' (Loop back with constructive feedback)
    """
    approved = state.get("critic_approved", False)
    iteration = state.get("iteration_count", 0)

    if approved or iteration >= settings.MAX_CRITIC_ITERATIONS:
        logger.info(
            "[Routing] Route -> 'assemble_result_node' (Approved=%s, Iteration=%d/%d)",
            approved,
            iteration,
            settings.MAX_CRITIC_ITERATIONS,
        )
        return "assemble_result_node"

    logger.info(
        "[Routing] Route -> 'copilot_agent_node' for revision (Iteration %d/%d). Feedback: %s",
        iteration,
        settings.MAX_CRITIC_ITERATIONS,
        state.get("critic_feedback", "Needs revision")[:80],
    )
    return "copilot_agent_node"


# ==============================================================================
# Graph Builder with Dependency Injection
# ==============================================================================

def build_copilot_graph(
    doc_parser: Optional[DocumentParser] = None,
    profile_parser: Optional[LLMProfileParser] = None,
    onet_repo: Optional[IOnetRepository] = None,
    career_service: Optional[Any] = None,
    evidence_retriever: Optional[IEvidenceRetriever] = None,
    copilot_llm: Optional[Any] = None,
    critic_llm: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
    interrupt_before: Optional[List[str]] = None,
) -> Any:
    """
    Builds, configures, and compiles the LangGraph Copilot StateGraph.

    Args:
        doc_parser: DocumentParser instance for reading PDF/DOCX/TXT
        profile_parser: LLMProfileParser instance for CandidateProfile extraction
        onet_repo: IOnetRepository instance for 0ms lookup and title matching
        career_service: UserCareerService for XGBoost and Clustering
        evidence_retriever: IEvidenceRetriever for RAG evidence
        copilot_llm: Injected LangChain LLM for Copilot (supports MockLLM)
        critic_llm: Injected LangChain LLM for Critic (supports MockLLM)
        checkpointer: LangGraph checkpointer (defaults to MemorySaver)
        interrupt_before: List of node names to interrupt before (defaults to ['human_review_node'])

    Returns:
        CompiledStateGraph: Executable state machine application.
    """
    logger.info("Assembling LangGraph Copilot StateGraph workflow...")
    workflow = StateGraph(AgentState)

    # 1. Define Node Closures injecting dependencies
    def _cv_ingest(state: AgentState) -> Dict[str, Any]:
        return cv_ingest_node(state, doc_parser=doc_parser)

    def _extract_profile(state: AgentState) -> Dict[str, Any]:
        return extract_profile_node(state, profile_parser=profile_parser, onet_repo=onet_repo)

    def _human_review(state: AgentState) -> Dict[str, Any]:
        return human_review_node(state)

    def _deterministic_lookup(state: AgentState) -> Dict[str, Any]:
        return deterministic_lookup(state, onet_repo=onet_repo)

    def _ml_scoring(state: AgentState) -> Dict[str, Any]:
        return ml_scoring_node(state, career_service=career_service)

    def _cluster_pivot(state: AgentState) -> Dict[str, Any]:
        return cluster_pivot_node(state, career_service=career_service)

    def _rag_grounding(state: AgentState) -> Dict[str, Any]:
        return rag_grounding_node(state, retriever=evidence_retriever)

    def _copilot_agent(state: AgentState) -> Dict[str, Any]:
        return copilot_agent_node(state, copilot_llm=copilot_llm)

    def _critic_evaluator(state: AgentState) -> Dict[str, Any]:
        return critic_evaluator_node(state, critic_llm=critic_llm)

    def _assemble_result(state: AgentState) -> Dict[str, Any]:
        return assemble_result_node(state)

    def _interactive_copilot(state: AgentState) -> Dict[str, Any]:
        return interactive_copilot_node(state, copilot_llm=copilot_llm, retriever=evidence_retriever)

    # 2. Add Nodes to Workflow
    workflow.add_node("cv_ingest_node", _cv_ingest)
    workflow.add_node("extract_profile_node", _extract_profile)
    workflow.add_node("human_review_node", _human_review)
    workflow.add_node("deterministic_lookup", _deterministic_lookup)
    workflow.add_node("ml_scoring_node", _ml_scoring)
    workflow.add_node("cluster_pivot_node", _cluster_pivot)
    workflow.add_node("rag_grounding_node", _rag_grounding)
    workflow.add_node("copilot_agent_node", _copilot_agent)
    workflow.add_node("critic_evaluator_node", _critic_evaluator)
    workflow.add_node("assemble_result_node", _assemble_result)
    workflow.add_node("interactive_copilot_node", _interactive_copilot)

    # 3. Add Edges & Conditional Branching
    workflow.add_conditional_edges(
        START,
        route_entrypoint,
        {
            "cv_ingest_node": "cv_ingest_node",
            "interactive_copilot_node": "interactive_copilot_node",
        },
    )

    workflow.add_edge("cv_ingest_node", "extract_profile_node")
    workflow.add_edge("extract_profile_node", "human_review_node")

    workflow.add_conditional_edges(
        "human_review_node",
        route_after_matcher,
        {
            "deterministic_lookup": "deterministic_lookup",
            "ml_scoring_node": "ml_scoring_node",
        },
    )

    workflow.add_edge("deterministic_lookup", "cluster_pivot_node")
    workflow.add_edge("ml_scoring_node", "cluster_pivot_node")
    workflow.add_edge("cluster_pivot_node", "rag_grounding_node")
    workflow.add_edge("rag_grounding_node", "copilot_agent_node")
    workflow.add_edge("copilot_agent_node", "critic_evaluator_node")

    workflow.add_conditional_edges(
        "critic_evaluator_node",
        route_after_critic,
        {
            "assemble_result_node": "assemble_result_node",
            "copilot_agent_node": "copilot_agent_node",
        },
    )

    workflow.add_edge("assemble_result_node", END)
    workflow.add_edge("interactive_copilot_node", END)

    # 4. Compile with Checkpointer and HITL interruption
    cp = checkpointer if checkpointer is not None else MemorySaver()
    interrupts = interrupt_before if interrupt_before is not None else ["human_review_node"]

    app = workflow.compile(
        checkpointer=cp,
        interrupt_before=interrupts,
    )

    logger.info("LangGraph Copilot StateGraph successfully compiled with checkpointer %s.", cp.__class__.__name__)
    return app
