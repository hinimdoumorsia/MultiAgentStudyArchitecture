"""
Shared state definition for the multi-agent academic system.
All agents read/write to this typed state via LangGraph's StateGraph.
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
import operator


class AgentResult(TypedDict):
    agent_name: str
    output: str
    confidence: float
    latency_ms: float
    success: bool
    metadata: Dict[str, Any]


# ================================================================
# 🔥 NOUVEAU CODE AJOUTÉ - COÛT ESTIMÉ DANS ROUTERDECISION 🔥
# ================================================================
class RouterDecision(TypedDict):
    selected_agents: List[str]
    reasoning: str
    estimated_complexity: str   # "low" | "medium" | "high"
    context_load: float         # 0.0 - 1.0
    # NOUVEAU: Coût estimé détaillé (tokens, temps, argent)
    estimated_cost: Dict[str, Any]  # {tokens, time_ms, monetary_cents, agents_needed}
# ================================================================
# FIN DE LA NOUVELLE PARTIE
# ================================================================


class AcademicState(TypedDict):
    # ── Core conversation ──────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    session_id: str

    # ── Orchestration ──────────────────────────────────────────────────
    router_decision: Optional[RouterDecision]
    plan: Optional[str]                    # from PlanningAgent
    retrieved_docs: Optional[str]          # from RAGAgent
    tool_results: str                      # from ToolsAgent
    verification_report: Optional[Dict]    # from VerificationAgent
    final_answer: Optional[str]            # from SynthesisAgent

    # ── Metrics (collected per run) ────────────────────────────────────
    agent_results: Annotated[List[AgentResult], operator.add]
    total_latency_ms: float
    run_id: str

    # ── Error handling ─────────────────────────────────────────────────
    errors: Annotated[List[str], operator.add]
    retry_count: int
    iteration_count: Annotated[int, operator.add]    # pour la boucle de vérification