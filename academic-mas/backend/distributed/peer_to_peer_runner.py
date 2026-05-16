"""
PeerToPeerRunner - Lance le pipeline P2P sans orchestrateur central.

Rôle : initialiser les agents, publier QUERY_RECEIVED, attendre SYNTHESIS_DONE.
Ce fichier ne contient AUCUNE logique de routage ou de décision métier.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

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

TIMEOUT_SECONDS = 180


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
        self._wrappers = [
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

        # 🔥 NOUVEAU : Extraire et additionner les tokens depuis agent_results
        agent_results = final_state.get("agent_results", [])
        tokens_total = 0
        tokens_prompt = 0
        tokens_compl = 0
        
        for agent in agent_results:
            agent_tokens = agent.get("tokens", {})
            tokens_total += agent_tokens.get("total_tokens", 0)
            tokens_prompt += agent_tokens.get("prompt_tokens", 0)
            tokens_compl += agent_tokens.get("completion_tokens", 0)

        # 🔥 NOUVEAU : Ajouter les tokens dans final_state pour qu'ils soient retournés
        final_state["tokens_total"] = tokens_total
        final_state["tokens_prompt"] = tokens_prompt
        final_state["tokens_completion"] = tokens_compl

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

        # ================================================================
        # 🔥 CORRECTION : Appel à memory_manager.record() avec TOUS les arguments
        # ================================================================
        try:
            memory_manager.record(
                session_id=session_id,
                run_id=run_id,
                query=query,
                answer=final_answer,
                agents_used=["planning", "rag", "tools", "verification", "synthesis"],
                confidence=confidence_score,
                latency_ms=total_latency_ms
            )
        except Exception as mem_exc:
            errors.append(f"Erreur mémoire : {mem_exc}")

        # ------------------------------------------------------------------
        # Format de retour IDENTIQUE à l'architecture hiérarchique
        # ------------------------------------------------------------------
        return {
            "session_id":          session_id,
            "run_id":              run_id,
            "query":               query,
            "final_answer":        final_answer,
            "plan":                final_state.get("plan", ""),
            "retrieved_docs":      final_state.get("retrieved_docs", ""),
            "tool_results":        final_state.get("tool_results", ""),
            "verification_report": verification_report,
            "router_decision":     final_state.get("router_decision", {}),
            "agent_results":       agent_results,
            "total_latency_ms":    total_latency_ms,
            "errors":              errors,
            "architecture":        "peer_to_peer",
            # 🔥 NOUVEAU : Ajouter les tokens dans le retour
            "tokens": {
                "total_tokens": tokens_total,
                "prompt_tokens": tokens_prompt,
                "completion_tokens": tokens_compl
            }
        }