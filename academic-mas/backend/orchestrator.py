"""
Orchestrator — LangGraph StateGraph

Architecture hiérarchique :
  router_node → planning → rag → tools → verification → synthesis → END

🔥 CORRECTIFS :
- `user_name` extrait depuis la query AVANT d'entrer dans le graphe
  et injecté dans le state initial → tous les agents y ont accès.
- `_extract_and_store_preferences` appelé IMMÉDIATEMENT sur la query
  entrante, avant même le premier nœud du graphe.
- Le `session_context` fourni au router inclut le nom si connu.
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
    registry.register(PlanningAgent())
    registry.register(RAGAgent())
    registry.register(ToolsAgent())
    registry.register(VerificationAgent())
    registry.register(SynthesisAgent())
    logger.info(f"[Orchestrator] Agents registered: {registry.list_agents()}")


def _router_node(state: AcademicState) -> Dict[str, Any]:
    """
    Entry node : calcule la décision de routage.
    user_query et user_name ne sont JAMAIS modifiés ici.
    """
    current_query = state["user_query"]
    session_ctx = memory_manager.get_session_context(state["session_id"])

    query_for_routing = current_query
    if session_ctx:
        query_for_routing = (
            f"[Contexte session]\n{session_ctx}\n\n"
            f"[Question actuelle]\n{current_query}"
        )

    decision = select_agents({**state, "user_query": query_for_routing})

    return {
        "router_decision": decision,
        "session_context": session_ctx,
        # user_query et user_name non retournés → inchangés dans le state
    }


def build_graph() -> StateGraph:
    graph = StateGraph(AcademicState)

    graph.add_node("router", _router_node)
    for name, agent in registry.all_agents().items():
        graph.add_node(name, agent)
    graph.add_node("skip", lambda s: {})

    graph.set_entry_point("router")

    agents = registry.list_agents()

    prev = "router"
    for step in ["planning", "rag", "tools", "verification", "synthesis"]:
        if step in agents:
            graph.add_edge(prev, step)
            prev = step

    graph.add_edge(prev if prev != "router" else "router", END) \
        if prev == "router" else graph.add_edge("synthesis", END) \
        if "synthesis" in agents else graph.add_edge(prev, END)

    return graph.compile()


def build_graph() -> StateGraph:
    graph = StateGraph(AcademicState)
    graph.add_node("router", _router_node)
    for name, agent in registry.all_agents().items():
        graph.add_node(name, agent)
    graph.add_node("skip", lambda s: {})
    graph.set_entry_point("router")

    agents = registry.list_agents()
    prev = "router"
    for step in ["planning", "rag", "tools", "verification"]:
        if step in agents:
            graph.add_edge(prev, step)
            prev = step

    if "synthesis" in agents:
        graph.add_edge(prev, "synthesis")
        graph.add_edge("synthesis", END)
    else:
        graph.add_edge(prev, END)

    return graph.compile()


class AcademicOrchestrator:

    def __init__(self):
        _register_default_agents()
        self._graph = build_graph()
        logger.info("[Orchestrator] Graph compiled successfully.")

    def run(self, query: str, session_id: str = None) -> Dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        start = time.perf_counter()

        # ─────────────────────────────────────────────────────────────────
        # 🔥 ÉTAPE 0 : Extraire le nom (et autres prefs) depuis la query
        #    AVANT de construire le state initial.
        #    Comme ça, si l'utilisateur dit "je me nomme X" dans ce tour,
        #    le state contient déjà user_name=X pour tous les agents.
        # ─────────────────────────────────────────────────────────────────
        memory_manager._extract_and_store_preferences(session_id, query)
        user_name = memory_manager.get_user_name(session_id)
        if user_name:
            logger.info(f"[Orchestrator] user_name résolu pour {session_id}: '{user_name}'")

        initial_state: AcademicState = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "session_id": session_id,
            "session_context": "",
            "router_decision": None,
            "plan": None,
            "retrieved_docs": None,
            "tool_results": "",
            "verification_report": None,
            "final_answer": "",
            "agent_results": [],
            "total_latency_ms": 0.0,
            "run_id": run_id,
            "errors": [],
            "retry_count": 0,
            "iteration_count": 0,
            # 🔥 user_name injecté ici — disponible pour TOUS les agents
            "user_name": user_name,
        }

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"[Orchestrator] Graph execution failed: {e}", exc_info=True)
            final_state = {
                **initial_state,
                "errors": [str(e)],
                "final_answer": f"Erreur système : {e}",
            }

        total_ms = (time.perf_counter() - start) * 1000
        final_state["total_latency_ms"] = round(total_ms, 2)

        # Filet de sécurité : uniquement si aucune réponse produite
        if not final_state.get("final_answer"):
            triggers = [
                "comment je m'appelle", "comment je me nomme",
                "quel est mon nom", "c'est quoi mon nom",
            ]
            if any(t in query.lower() for t in triggers):
                if user_name:
                    final_state["final_answer"] = f"🎯 Vous vous appelez **{user_name}** !"
                else:
                    final_state["final_answer"] = (
                        "Je ne connais pas encore votre nom. "
                        "Dites-le moi avec 'je me nomme …' ou 'je m'appelle …'."
                    )

        # Persist
        agents_used = [r["agent_name"] for r in final_state.get("agent_results", [])]
        verification = final_state.get("verification_report") or {}
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
        self._graph = build_graph()
        logger.info("[Orchestrator] Graph rebuilt.")

    def get_registered_agents(self) -> Dict[str, str]:
        return registry.agent_descriptions()