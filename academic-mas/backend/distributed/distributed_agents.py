"""
DistributedAgentWrapper - Encapsule chaque agent existant sans toucher à sa logique métier.

Chaque wrapper :
  1. S'abonne aux bons événements via subscribe_to_events()
  2. À la réception d'un événement, reconstruit un AcademicState depuis l'EventBus
  3. Appelle agent.process(state) INCHANGÉ
  4. Publie le résultat comme nouvel événement
"""

from __future__ import annotations

import time
import threading
import traceback
from typing import Any, Dict, Set

# ── Imports de l'infrastructure existante ────────────────────────────────────
from backend.agents.planning_agent import PlanningAgent
from backend.agents.rag_agent import RAGAgent
from backend.agents.tools_agent import ToolsAgent
from backend.agents.verification_agent import VerificationAgent
from backend.agents.synthesis_agent import SynthesisAgent
from backend.state import AcademicState
from backend.distributed.event_bus import EventBus, Event, EventType

# Instance globale du bus (singleton)
event_bus = EventBus()


def _build_state(session_id: str, query: str, bus_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruit un dictionnaire AcademicState complet à partir de l'état courant du bus.
    Retourne un dict compatible avec AcademicState (TypedDict).
    """
    return {
        "session_id": session_id,
        "user_query": query,
        "messages": [],
        "router_decision": bus_state.get("router_decision", {}),
        "plan": bus_state.get("plan", ""),
        "retrieved_docs": bus_state.get("retrieved_docs", ""),
        "tool_results": bus_state.get("tool_results", ""),
        "verification_report": bus_state.get("verification_report", {}),
        "final_answer": bus_state.get("final_answer", ""),
        "agent_results": bus_state.get("agent_results", []),
        "total_latency_ms": 0.0,
        "run_id": "",
        "errors": bus_state.get("errors", []),
        "retry_count": 0,
        "iteration_count": 0
    }


class DistributedAgentWrapper:
    """
    Wrapper générique : prend un agent existant, ne touche pas à process().
    Sous-classes concrètes définissent :
      - trigger_events  : liste d'EventType auxquels s'abonner
      - output_event    : EventType à publier après process()
      - _extract_payload: comment extraire le payload depuis le résultat de process()
    """

    trigger_events: list[EventType] = []
    output_event: EventType = EventType.ERROR

    # ================================================================
    # 🔥 CORRECTION : __init__ accepte event_bus
    # ================================================================
    def __init__(self, agent: Any, event_bus: EventBus = None) -> None:
        self.agent = agent
        self.bus = event_bus or EventBus()  # Utilise le bus passé ou le singleton
        self._ran: Set[str] = set()  # Pour éviter double exécution
        self._lock = threading.Lock()
        # 🔥 NOUVEAU : Pour tracker le dernier appel de chaque agent
        self._last_call_time: Dict[str, float] = {}  # session_id -> timestamp

    def subscribe_to_events(self) -> None:
        for event_type in self.trigger_events:
            self.bus.subscribe(event_type, self._on_event)

    def _has_run(self, session_id: str) -> bool:
        """Vérifie si l'agent a déjà été exécuté pour cette session"""
        with self._lock:
            return session_id in self._ran
    
    def _mark_run(self, session_id: str) -> None:
        """Marque l'agent comme exécuté pour cette session"""
        with self._lock:
            self._ran.add(session_id)
    
    # 🔥 NOUVEAU : Méthode pour espacer les appels
    def _throttle_if_needed(self, session_id: str) -> None:
        """Espace les appels au LLM pour éviter les rate limits"""
        with self._lock:
            now = time.time()
            last_call = self._last_call_time.get(session_id, 0)
            
            # Attendre 2 secondes entre chaque appel pour la même session
            min_interval = 5.0  # 2 secondes entre les appels
            time_since_last = now - last_call
            
            if time_since_last < min_interval and last_call > 0:
                wait_time = min_interval - time_since_last
                print(f"⏳ [{self.agent.__class__.__name__}] Attente de {wait_time:.2f}s pour espacer les appels LLM...")
                time.sleep(wait_time)
            
            self._last_call_time[session_id] = time.time()

    def _extract_output(self, result: Dict) -> str:
        """Extrait le texte de sortie principal du résultat de l'agent."""
        if result is None:
            return ""
        for key in ("final_answer", "plan", "retrieved_docs", "tool_results"):
            if result.get(key):
                v = result[key]
                return v if isinstance(v, str) else str(v)
        return ""

    def _extract_confidence(self, result: Dict) -> float:
        """Extrait le score de confiance du résultat."""
        if result is None:
            return 0.85
        report = result.get("verification_report", {})
        if report:
            return report.get("confidence_score", 0.8)
        return 0.85

    def _build_agent_result(self, result: Dict) -> Dict:
        """Construit l'entrée agent_results à partir du résultat de process()."""
        return {
            "agent_name": self.agent.name,
            "output": self._extract_output(result),
            "confidence": self._extract_confidence(result),
            "tokens": result.get("tokens", {}),
            "success": True,
            "latency_ms": 0  # Sera mis à jour par BaseAgent normalement
        }

    def _on_event(self, event: Event) -> None:
        session_id = event.session_id
        
        # Éviter double exécution
        if self._has_run(session_id):
            return
        self._mark_run(session_id)
        
        # 🔥 NOUVEAU : Espacer les appels avant de traiter
        self._throttle_if_needed(session_id)
        
        query = event.payload.get("user_query", "")

        try:
            bus_state = self.bus.get_state(session_id)
            state = _build_state(session_id, query, bus_state)

            # ── Appel INCHANGÉ de la logique métier ──────────────────────────
            result = self.agent.process(state)
            # ─────────────────────────────────────────────────────────────────

            # 🔥 NOUVEAU : Construire l'agent_result
            agent_result = self._build_agent_result(result)
            
            payload = self._extract_payload(result, state)
            payload["user_query"] = query
            # 🔥 AJOUTER agent_results au payload
            payload["agent_results"] = [agent_result]

            self.bus.publish(Event(
                type=self.output_event,
                payload=payload,
                source=self.agent.__class__.__name__,
                session_id=session_id,
            ))

        except Exception as exc:
            tb = traceback.format_exc()
            self.bus.publish(Event(
                type=EventType.ERROR,
                payload={
                    "user_query": query,
                    "error": str(exc),
                    "traceback": tb,
                    "agent": self.agent.__class__.__name__,
                    "agent_results": [{
                        "agent_name": self.agent.name,
                        "output": "",
                        "confidence": 0.0,
                        "tokens": {},
                        "success": False
                    }]
                },
                source=self.agent.__class__.__name__,
                session_id=session_id,
            ))

    def _extract_payload(self, result: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """À surcharger dans les sous-classes si nécessaire."""
        return result or {}


# ══════════════════════════════════════════════════════════════════════════════
# Wrappers concrets — un par agent
# ══════════════════════════════════════════════════════════════════════════════

class DistributedPlanningAgent(DistributedAgentWrapper):
    """
    Déclenché par : QUERY_RECEIVED
    Publie        : PLAN_CREATED
    """
    trigger_events = [EventType.QUERY_RECEIVED]
    output_event   = EventType.PLAN_CREATED

    def _extract_payload(self, result: Dict, state: Dict) -> Dict:
        return {
            "plan": result.get("plan", ""),
            "router_decision": result.get("router_decision", {}),
            "tokens": result.get("tokens", {}),
        }


class DistributedRAGAgent(DistributedAgentWrapper):
    """
    Déclenché par : PLAN_CREATED
    Publie        : DOCUMENTS_FOUND
    """
    trigger_events = [EventType.PLAN_CREATED]
    output_event   = EventType.DOCUMENTS_FOUND

    def _extract_payload(self, result: Dict, state: Dict) -> Dict:
        return {
            "retrieved_docs": result.get("retrieved_docs", ""),
            "tokens": result.get("tokens", {}),
        }


class DistributedToolsAgent(DistributedAgentWrapper):
    """
    Déclenché par : PLAN_CREATED et DOCUMENTS_FOUND
    Publie        : TOOL_EXECUTED
    """
    trigger_events = [EventType.PLAN_CREATED, EventType.DOCUMENTS_FOUND]
    output_event   = EventType.TOOL_EXECUTED

    def _extract_payload(self, result: Dict, state: Dict) -> Dict:
        return {
            "tool_results": result.get("tool_results", ""),
            "tokens": result.get("tokens", {}),
        }


class DistributedVerificationAgent(DistributedAgentWrapper):
    """
    Déclenché par : TOOL_EXECUTED
    Publie        : VERIFICATION_DONE
    """
    trigger_events = [EventType.TOOL_EXECUTED]
    output_event   = EventType.VERIFICATION_DONE

    def _extract_payload(self, result: Dict, state: Dict) -> Dict:
        return {
            "verification_report": result.get("verification_report", {}),
            "tokens": result.get("tokens", {}),
        }


class DistributedSynthesisAgent(DistributedAgentWrapper):
    """
    Déclenché par : VERIFICATION_DONE
    Publie        : SYNTHESIS_DONE
    """
    trigger_events = [EventType.VERIFICATION_DONE]
    output_event   = EventType.SYNTHESIS_DONE

    def _extract_payload(self, result: Dict, state: Dict) -> Dict:
        return {
            "final_answer": result.get("final_answer", ""),
            "tokens": result.get("tokens", {}),
        }