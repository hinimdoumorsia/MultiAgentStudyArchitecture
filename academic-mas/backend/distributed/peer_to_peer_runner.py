"""
PeerToPeerRunner - Lance le pipeline P2P sans orchestrateur central.

Rôle : initialiser les agents, publier QUERY_RECEIVED, attendre SYNTHESIS_DONE.
Ce fichier ne contient AUCUNE logique de routage ou de décision métier.
Version alignée avec le format de réponse hiérarchique.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional, List

# ── Agents existants ──────────────────────────────────────────────────────────
from backend.agents.planning_agent import PlanningAgent
from backend.agents.rag_agent import RAGAgent
from backend.agents.tools_agent import ToolsAgent
from backend.agents.verification_agent import VerificationAgent
from backend.agents.synthesis_agent import SynthesisAgent

# ── Infrastructure mémoire existante ─────────────────────────────────────────
from backend.memory.memory_manager import memory_manager

# ── Bus et wrappers ───────────────────────────────────────────────────────────
from backend.distributed.event_bus import EventBus, Event, EventType
from backend.distributed.distributed_agents import (
    DistributedPlanningAgent,
    DistributedRAGAgent,
    DistributedToolsAgent,
    DistributedVerificationAgent,
    DistributedSynthesisAgent,
)

TIMEOUT_SECONDS = 300  # Passage à 5 minutes pour plus de robustesse


class PeerToPeerRunner:
    """
    Point d'entrée du pipeline P2P.
    - Instancie les agents UNE SEULE FOIS (singleton-like pour le serveur).
    - Publie QUERY_RECEIVED et attend SYNTHESIS_DONE.
    - Retourne un dict au MÊME format que l'architecture hiérarchique.
    """

    def __init__(self) -> None:
        self.bus = EventBus()

        # Instanciation des agents existants
        planning_agent     = PlanningAgent()
        rag_agent          = RAGAgent()
        tools_agent        = ToolsAgent()
        verification_agent = VerificationAgent()
        synthesis_agent    = SynthesisAgent()

        # Encapsulation dans les wrappers distribués
        self._wrappers: List[Any] = [
            DistributedPlanningAgent(planning_agent, self.bus),
            DistributedRAGAgent(rag_agent, self.bus),
            DistributedToolsAgent(tools_agent, self.bus),
            DistributedVerificationAgent(verification_agent, self.bus),
            DistributedSynthesisAgent(synthesis_agent, self.bus),
        ]

        # Abonnements : chaque wrapper s'enregistre sur le bus
        for wrapper in self._wrappers:
            wrapper.subscribe_to_events()

    # ------------------------------------------------------------------
    # Point d'entrée public
    # ------------------------------------------------------------------

    def run(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lance le pipeline et attend la fin.
        Retourne un dict compatible avec l'endpoint hiérarchique.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        run_id    = str(uuid.uuid4())
        t_start   = time.time()

        # Prépare le threading.Event pour attendre sans polling
        done_event = self.bus.register_session(session_id)

        # Démarre le pipeline en publiant le premier événement
        self.bus.publish(Event(
            type=EventType.QUERY_RECEIVED,
            payload={"user_query": query},
            source="PeerToPeerRunner",
            session_id=session_id,
        ))

        # Attend SYNTHESIS_DONE ou timeout
        finished = done_event.wait(timeout=TIMEOUT_SECONDS)
        total_latency_ms = (time.time() - t_start) * 1000

        # Récupère l'état final
        final_state = self.bus.get_state(session_id)

        # ================================================================
        # 🔥 EXTRACTION DES AGENT_RESULTS AU MÊME FORMAT QUE HIÉRARCHIQUE
        # ================================================================
        agent_results = final_state.get("agent_results", [])
        
        # Si agent_results est vide, on le construit à partir des résultats individuels
        if not agent_results:
            agent_results = []
            for wrapper in self._wrappers:
                result = final_state.get(f"{wrapper.agent_name}_result", {})
                if result:
                    agent_results.append({
                        "agent_name": wrapper.agent_name,
                        "output": result.get("output", ""),
                        "confidence": result.get("confidence", 0.85),
                        "latency_ms": result.get("latency_ms", 0),
                        "success": result.get("success", True),
                        "metadata": result.get("metadata", {}),
                        "tokens": result.get("tokens", {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        })
                    })

        # ================================================================
        # 🔥 CALCUL DES TOKENS TOTAUX (comme dans l'architecture hiérarchique)
        # ================================================================
        tokens_total = 0
        tokens_prompt = 0
        tokens_compl = 0
        
        for agent in agent_results:
            agent_tokens = agent.get("tokens", {})
            tokens_total += agent_tokens.get("total_tokens", 0)
            tokens_prompt += agent_tokens.get("prompt_tokens", 0)
            tokens_compl += agent_tokens.get("completion_tokens", 0)

        # ================================================================
        # 🔥 GARANTIR QUE CHAQUE AGENT A UN CHAMP "tokens"
        # ================================================================
        for agent in agent_results:
            if "tokens" not in agent or not agent["tokens"]:
                agent["tokens"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }

        # ================================================================
        # 🔥 RÉCUPÉRATION DE LA RÉPONSE FINALE
        # ================================================================
        final_answer = final_state.get("final_answer", "")
        errors: list = []

        if not finished or not self.bus.is_done(session_id):
            errors.append("Timeout : le pipeline n'a pas terminé dans le délai imparti.")
            if not final_answer:
                final_answer = "Erreur : timeout du pipeline P2P."

        if "error" in final_state:
            errors.append(final_state["error"])
            if not final_answer:
                final_answer = f"Erreur : {final_state['error']}"

        # Récupérer le rapport de vérification pour le score de confiance
        verification_report = final_state.get("verification_report", {})
        confidence_score = verification_report.get("confidence_score", 0.8)

        # Récupérer la décision du routeur (si disponible)
        router_decision = final_state.get("router_decision", {
            "selected_agents": ["planning", "rag", "tools", "verification", "synthesis"],
            "reasoning": "architecture distribuée peer-to-peer",
            "estimated_complexity": "medium",
            "context_load": 0.5
        })

        # ================================================================
        # 🔥 ENREGISTREMENT EN MÉMOIRE (comme hiérarchique)
        # ================================================================
        try:
            memory_manager.record(
                session_id=session_id,
                run_id=run_id,
                query=query,
                answer=final_answer,
                agents_used=[a.get("agent_name", "unknown") for a in agent_results],
                confidence=confidence_score,
                latency_ms=total_latency_ms
            )
        except Exception as mem_exc:
            errors.append(f"Erreur mémoire : {mem_exc}")

        # ================================================================
        # 🔥 FORMAT DE RETOUR IDENTIQUE À L'ARCHITECTURE HIÉRARCHIQUE
        # ================================================================
        return {
            "session_id": session_id,
            "run_id": run_id,
            "query": query,
            "final_answer": final_answer,
            "plan": final_state.get("plan", ""),
            "retrieved_docs": final_state.get("retrieved_docs", ""),
            "tool_results": final_state.get("tool_results", ""),
            "verification_report": verification_report,
            "router_decision": router_decision,
            "agent_results": agent_results,
            "total_latency_ms": total_latency_ms,
            "errors": errors,
            "architecture": "peer_to_peer"
        }