"""
Orchestrator — LangGraph StateGraph

Construit le graphe d'exécution multi-agents de façon DYNAMIQUE :
- Les nœuds sont ajoutés depuis le registry
- Les edges conditionnels lisent la RouterDecision
- Ajouter un agent = juste l'enregistrer, le graphe s'adapte

Architecture hiérarchique :
  router_node → planning → [rag, tools] → verification → synthesis → END
"""

import time
import uuid
import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from backend.state import AcademicState
from backend.agents.registry import registry
from backend.agents.planning_agent import PlanningAgent
from backend.agents.rag_agent import RAGAgent
from backend.agents.tools_agent import ToolsAgent
from backend.agents.verification_agent import VerificationAgent
from backend.agents.synthesis_agent import SynthesisAgent
from backend.memory.memory_manager import memory_manager
from backend.utils.router import select_agents

logger = logging.getLogger(__name__)


def _register_default_agents():
    """Register the default agent set. Call once at startup."""
    registry.register(PlanningAgent())
    registry.register(RAGAgent())
    registry.register(ToolsAgent())
    registry.register(VerificationAgent())
    registry.register(SynthesisAgent())
    logger.info(f"[Orchestrator] Agents registered: {registry.list_agents()}")


def _router_node(state: AcademicState) -> Dict[str, Any]:
    """Entry node: computes routing decision."""
    session_ctx = memory_manager.get_session_context(state["session_id"])
    query = state["user_query"]
    if session_ctx:
        query = f"[Contexte session]\n{session_ctx}\n\n[Question actuelle]\n{query}"

    decision = select_agents({**state, "user_query": query})
    return {"router_decision": decision}


def build_graph() -> StateGraph:
    """
    Build graph with NO iterative verification loop (fixé à 1 passage).
    Structure: router → planning → rag → tools → verification → synthesis → END
    """
    graph = StateGraph(AcademicState)

    # Nodes
    graph.add_node("router", _router_node)
    
    # Register all agents
    for name, agent in registry.all_agents().items():
        graph.add_node(name, agent)
    
    graph.add_node("skip", lambda s: {})

    # Entry point
    graph.set_entry_point("router")

    agents = registry.list_agents()
    
    # router → planning
    if "planning" in agents:
        graph.add_edge("router", "planning")
        prev = "planning"
    else:
        prev = "router"

    # planning → rag
    if "rag" in agents:
        graph.add_edge(prev, "rag")
        prev = "rag"

    # rag → tools
    if "tools" in agents:
        graph.add_edge(prev, "tools")
        prev = "tools"

    # tools → verification
    if "verification" in agents:
        graph.add_edge(prev, "verification")
        prev = "verification"

    # verification → synthesis → END (PAS DE BOUCLE)
    if "synthesis" in agents:
        graph.add_edge(prev, "synthesis")
        graph.add_edge("synthesis", END)
    else:
        graph.add_edge(prev, END)

    return graph.compile()


class AcademicOrchestrator:
    """High-level interface to the multi-agent system."""

    def __init__(self):
        _register_default_agents()
        self._graph = build_graph()
        logger.info("[Orchestrator] Graph compiled successfully.")

    def run(self, query: str, session_id: str = None) -> Dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        start = time.perf_counter()

        # État initial avec toutes les valeurs en string, pas None
        initial_state: AcademicState = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "session_id": session_id,
            "router_decision": None,
            "plan": None,
            "retrieved_docs": None,
            "tool_results": "",  # string vide
            "verification_report": None,
            "final_answer": "",  # string vide au lieu de None
            "agent_results": [],
            "total_latency_ms": 0.0,
            "run_id": run_id,
            "errors": [],
            "retry_count": 0,
            "iteration_count": 0,
        }

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"[Orchestrator] Graph execution failed: {e}", exc_info=True)
            final_state = {**initial_state, "errors": [str(e)], "final_answer": f"Erreur système : {e}"}

        total_ms = (time.perf_counter() - start) * 1000
        final_state["total_latency_ms"] = round(total_ms, 2)

        # Persist to memory
        agents_used = [r["agent_name"] for r in final_state.get("agent_results", [])]
        verification = final_state.get("verification_report")
        if verification is None:
            verification = {}
        confidence = verification.get("confidence_score", 0.8)

        memory_manager.record(
            session_id=session_id,
            run_id=run_id,
            query=query,
            answer=final_state.get("final_answer", ""),
            agents_used=agents_used,
            confidence=confidence,
            latency_ms=total_ms,
        )

        return {
            "session_id": session_id,
            "run_id": run_id,
            "query": query,
            "final_answer": final_state.get("final_answer", "Aucune réponse générée"),
            "plan": final_state.get("plan"),
            "retrieved_docs": final_state.get("retrieved_docs"),
            "tool_results": final_state.get("tool_results"),
            "verification_report": final_state.get("verification_report"),
            "router_decision": final_state.get("router_decision"),
            "agent_results": final_state.get("agent_results", []),
            "total_latency_ms": final_state.get("total_latency_ms"),
            "errors": final_state.get("errors", []),
        }

    def rebuild_graph(self):
        """Hot-reload graph after adding/removing agents from registry."""
        self._graph = build_graph()
        logger.info("[Orchestrator] Graph rebuilt.")

    def get_registered_agents(self) -> Dict[str, str]:
        return registry.agent_descriptions()