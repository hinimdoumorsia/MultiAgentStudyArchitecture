"""
EventBus - Infrastructure de communication peer-to-peer.
Ne contient AUCUNE logique métier. Juste le bus d'événements.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    QUERY_RECEIVED    = "QUERY_RECEIVED"
    PLAN_CREATED      = "PLAN_CREATED"
    DOCUMENTS_FOUND   = "DOCUMENTS_FOUND"
    TOOL_EXECUTED     = "TOOL_EXECUTED"
    VERIFICATION_DONE = "VERIFICATION_DONE"
    SYNTHESIS_DONE    = "SYNTHESIS_DONE"
    ERROR             = "ERROR"


@dataclass
class Event:
    type: EventType
    payload: Dict[str, Any]
    source: str                          # nom de l'agent émetteur
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EventBus:
    """
    Singleton thread-safe.
    Les agents s'abonnent à des types d'événements et publient leurs résultats.
    """

    _instance: Optional["EventBus"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EventBus":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        # { EventType -> [callback, ...] }
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = defaultdict(list)
        # { session_id -> { clé -> valeur } }  état agrégé par session
        self._states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        # { session_id -> bool }
        self._done: Dict[str, bool] = defaultdict(bool)
        # { session_id -> threading.Event }  pour le wait() dans le runner
        self._done_events: Dict[str, threading.Event] = {}
        self._state_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Abonnement
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Enregistre un callback pour un type d'événement donné."""
        self._subscribers[event_type].append(callback)

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------

    def _update_agent_results(self, session_id: str, payload: Dict[str, Any]) -> None:
        """
        Accumule les agent_results dans l'état sans écraser.
        """
        if "agent_results" in payload:
            existing = self._states[session_id].get("agent_results", [])
            new_results = payload["agent_results"]
            if isinstance(new_results, list):
                self._states[session_id]["agent_results"] = existing + new_results
            # Supprimer agent_results du payload pour ne pas écraser avec update()
            del payload["agent_results"]

    def publish(self, event: Event) -> None:
        """
        Publie un événement :
        1. Met à jour l'état de la session avec le payload.
        2. Accumule les agent_results séparément.
        3. Marque la session comme terminée si SYNTHESIS_DONE.
        4. Notifie tous les abonnés de façon asynchrone.
        """
        with self._state_lock:
            # Copier le payload pour ne pas modifier l'original
            payload = event.payload.copy()
            
            # Gérer les agent_results séparément (accumulation)
            self._update_agent_results(event.session_id, payload)
            
            # Mettre à jour l'état avec le reste du payload
            self._states[event.session_id].update(payload)
            
            if event.type == EventType.SYNTHESIS_DONE:
                self._done[event.session_id] = True
                done_evt = self._done_events.get(event.session_id)
                if done_evt:
                    done_evt.set()
            if event.type == EventType.ERROR:
                # Une erreur bloque aussi le wait() pour éviter le timeout
                done_evt = self._done_events.get(event.session_id)
                if done_evt:
                    done_evt.set()

        # Notification dans des threads séparés pour ne pas bloquer le publisher
        for callback in self._subscribers.get(event.type, []):
            t = threading.Thread(target=self._safe_call, args=(callback, event), daemon=True)
            t.start()

    @staticmethod
    def _safe_call(callback: Callable[[Event], None], event: Event) -> None:
        try:
            callback(event)
        except Exception as exc:
            print(f"[EventBus] Erreur dans le callback {callback} : {exc}")

    # ------------------------------------------------------------------
    # Lecture d'état
    # ------------------------------------------------------------------

    def get_state(self, session_id: str) -> Dict[str, Any]:
        """Retourne une copie de l'état agrégé de la session."""
        with self._state_lock:
            return dict(self._states[session_id])

    def is_done(self, session_id: str) -> bool:
        """Retourne True si SYNTHESIS_DONE a été publié pour cette session."""
        return self._done.get(session_id, False)

    # ------------------------------------------------------------------
    # Utilitaires pour le runner
    # ------------------------------------------------------------------

    def register_session(self, session_id: str) -> threading.Event:
        """
        Crée un threading.Event pour permettre au runner d'attendre
        la fin du pipeline sans polling actif.
        """
        evt = threading.Event()
        with self._state_lock:
            self._done_events[session_id] = evt
        return evt

    def reset(self) -> None:
        """Réinitialise le bus (utile pour les tests)."""
        with self._state_lock:
            self._subscribers.clear()
            self._states.clear()
            self._done.clear()
            self._done_events.clear()