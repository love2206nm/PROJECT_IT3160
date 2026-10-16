"""
LangGraph Multi-Agent Copilot Module.
Provides stateful career advisory state machine with self-correcting critic loops.
"""

from source.model.LLM_model.copilot.state import AgentState
from source.model.LLM_model.copilot.graph import (
    build_copilot_graph,
    route_entrypoint,
    route_after_matcher,
    route_after_critic,
)
from source.model.LLM_model.copilot.prompts import (
    CriticEvaluation,
    CriticScores,
    COPILOT_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
)
from source.model.LLM_model.copilot.workflow import LangGraphCopilotWorkflow

__all__ = [
    "AgentState",
    "build_copilot_graph",
    "LangGraphCopilotWorkflow",
    "route_entrypoint",
    "route_after_matcher",
    "route_after_critic",
    "CriticEvaluation",
    "CriticScores",
    "COPILOT_SYSTEM_PROMPT",
    "CRITIC_SYSTEM_PROMPT",
]
