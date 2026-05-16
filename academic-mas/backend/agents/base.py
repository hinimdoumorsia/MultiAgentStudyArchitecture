"""
BaseAgent — every specialized agent inherits from this class.

To ADD a new agent:
    1. Create backend/agents/my_agent.py
    2. Subclass BaseAgent, implement `process()`
    3. Register it in AgentRegistry (registry.py)
    4. Add a route in orchestrator.py

To REMOVE an agent:
    1. Unregister it from AgentRegistry
    2. Remove its route from the graph
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import time
import logging

from backend.state import AcademicState, AgentResult

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all agents in the hierarchy.

    Attributes
    ----------
    name : str
        Unique identifier used by the router and registry.
    description : str
        Human-readable role description (shown in UI + README).
    """

    name: str = "base_agent"
    description: str = "Abstract base agent"

    def __call__(self, state: AcademicState) -> Dict[str, Any]:
        """LangGraph-compatible node: wraps process() with timing + error handling."""
        start = time.perf_counter()
        try:
            logger.info(f"[{self.name}] Processing query: {state['user_query'][:80]}…")
            result_state = self.process(state)
            
            # Si result_state est None, on le remplace par un dict vide
            if result_state is None:
                result_state = {}
            
            latency_ms = (time.perf_counter() - start) * 1000

            # 🔥 NOUVEAU : Extraire les tokens
            tokens = self._extract_tokens(result_state)

            agent_result: AgentResult = {
                "agent_name": self.name,
                "output": self._extract_output(result_state),
                "confidence": self._extract_confidence(result_state),
                "latency_ms": round(latency_ms, 2),
                "success": True,
                "metadata": self.get_metadata(state),
                "tokens": tokens,  # ← AJOUT DES TOKENS
            }
            
            # Conserver les clés existantes
            if "agent_results" not in result_state:
                result_state["agent_results"] = []
            result_state["agent_results"].append(agent_result)
            
            logger.info(f"[{self.name}] Done in {latency_ms:.0f}ms, tokens: {tokens.get('total_tokens', 0)}")
            return result_state

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[{self.name}] Error: {exc}", exc_info=True)
            
            # Retourner un état complet qui préserve les données existantes
            return {
                "errors": [f"{self.name}: {str(exc)}"],
                "agent_results": [{
                    "agent_name": self.name,
                    "output": "",
                    "confidence": 0.0,
                    "latency_ms": round(latency_ms, 2),
                    "success": False,
                    "metadata": {},
                    "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},  # ← AJOUT
                }],
                # Préserver les clés existantes pour éviter None
                "plan": state.get("plan", ""),
                "retrieved_docs": state.get("retrieved_docs", ""),
                "tool_results": state.get("tool_results", ""),
                "verification_report": state.get("verification_report", {}),
                "final_answer": state.get("final_answer", ""),
                "user_query": state.get("user_query", ""),
                "messages": state.get("messages", []),
                "session_id": state.get("session_id", ""),
                "router_decision": state.get("router_decision", {}),
                "total_latency_ms": state.get("total_latency_ms", 0.0),
                "run_id": state.get("run_id", ""),
                "retry_count": state.get("retry_count", 0),
            }

    @abstractmethod
    def process(self, state: AcademicState) -> Dict[str, Any]:
        """
        Core agent logic. Receives full state, returns a partial state dict
        with only the keys this agent updates.
        """
        ...

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        """Override to expose agent-specific metadata to the UI."""
        return {}

    # ── Internal helpers ───────────────────────────────────────────────
    def _extract_output(self, result_state: Dict) -> str:
        if result_state is None:
            return ""
        for key in ("final_answer", "plan", "retrieved_docs", "tool_results"):
            if result_state.get(key):
                v = result_state[key]
                return v if isinstance(v, str) else str(v)
        return ""

    def _extract_confidence(self, result_state: Dict) -> float:
        if result_state is None:
            return 0.85
        report = result_state.get("verification_report", {})
        if report:
            return report.get("confidence_score", 0.8)
        return 0.85

    def _extract_tokens(self, result_state: Dict) -> Dict[str, int]:
        """
        Extrait les tokens du résultat de l'agent.
        À utiliser dans chaque agent qui appelle un LLM.
        """
        tokens = result_state.get("tokens", {})
        return {
            "prompt_tokens": tokens.get("prompt_tokens", 0),
            "completion_tokens": tokens.get("completion_tokens", 0),
            "total_tokens": tokens.get("total_tokens", 0)
        }