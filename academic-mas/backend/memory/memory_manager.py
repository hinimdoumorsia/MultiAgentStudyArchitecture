"""
MemoryManager — Gestion de la mémoire à deux niveaux

- Mémoire de session : contexte de la conversation en cours (in-memory)
- Mémoire persistante : historique inter-sessions (SQLite)
"""

import sqlite3
import json
import os
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionMemory:
    """In-memory session store. Cleared on restart."""

    def __init__(self):
        self._sessions: Dict[str, List[Dict]] = {}

    def add_turn(self, session_id: str, query: str, answer: str, metadata: Dict = None):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({
            "timestamp": time.time(),
            "query": query,
            "answer": answer,
            "metadata": metadata or {},
        })
        # Keep last 20 turns per session
        self._sessions[session_id] = self._sessions[session_id][-20:]

    def get_history(self, session_id: str) -> List[Dict]:
        return self._sessions.get(session_id, [])

    def get_context_string(self, session_id: str, last_n: int = 3) -> str:
        history = self.get_history(session_id)[-last_n:]
        if not history:
            return ""
        lines = []
        for turn in history:
            lines.append(f"Q: {turn['query'][:200]}")
            lines.append(f"A: {turn['answer'][:300]}")
        return "\n".join(lines)

    def clear_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())


class PersistentMemory:
    """SQLite-backed persistent memory. Survives restarts."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("MEMORY_DB_PATH", "./data/memory.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    run_id TEXT,
                    query TEXT NOT NULL,
                    answer TEXT,
                    agents_used TEXT,
                    confidence REAL,
                    latency_ms REAL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON conversations(session_id)
            """)
            conn.commit()
        logger.info(f"[PersistentMemory] DB ready at {self.db_path}")

    def save_conversation(
        self,
        session_id: str,
        run_id: str,
        query: str,
        answer: str,
        agents_used: List[str],
        confidence: float,
        latency_ms: float,
        metadata: Dict = None,
    ):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversations
                (session_id, run_id, query, answer, agents_used,
                 confidence, latency_ms, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, run_id, query, answer,
                json.dumps(agents_used),
                confidence, latency_ms,
                datetime.now().isoformat(),
                json.dumps(metadata or {}),
            ))
            conn.commit()

    def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT query, answer, agents_used, confidence, latency_ms, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
            """, (session_id, limit)).fetchall()
        return [
            {
                "query": r[0], "answer": r[1],
                "agents_used": json.loads(r[2] or "[]"),
                "confidence": r[3], "latency_ms": r[4], "timestamp": r[5],
            }
            for r in reversed(rows)
        ]

    def get_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM conversations").fetchone()[0]
            avg_lat = conn.execute("SELECT AVG(latency_ms) FROM conversations").fetchone()[0]
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM conversations"
            ).fetchone()[0]
        return {
            "total_conversations": total,
            "avg_confidence": round(avg_conf or 0, 3),
            "avg_latency_ms": round(avg_lat or 0, 1),
            "total_sessions": sessions,
        }


class MemoryManager:
    """Facade combining session + persistent memory."""

    def __init__(self):
        self.session = SessionMemory()
        self.persistent = PersistentMemory()

    def record(self, session_id: str, run_id: str, query: str, answer: str,
               agents_used: List[str], confidence: float, latency_ms: float):
        """Enregistre une conversation dans les deux mémoires"""
        self.session.add_turn(session_id, query, answer)
        self.persistent.save_conversation(
            session_id, run_id, query, answer,
            agents_used, confidence, latency_ms,
        )

    # ================================================================
    # 🔥 NOUVEAU CODE AJOUTÉ - CORRECTION DE LA MÉMOIRE PERSISTANTE 🔥
    # ================================================================
    def get_session_context(self, session_id: str, last_n: int = 3) -> str:
        """
        Récupère le contexte d'une session.
        
        🔥 AMÉLIORATION : Maintenant, si la mémoire session (RAM) est vide
        (par exemple après un redémarrage du serveur), cette fonction va
        automatiquement charger l'historique depuis la mémoire persistante
        (SQLite) et le restaurer dans la mémoire session.
        
        Cela permet à l'agent de se souvenir des conversations précédentes
        même après un redémarrage du serveur !
        """
        
        # ÉTAPE 1 : D'abord essayer de récupérer depuis SessionMemory (RAM)
        # C'est plus rapide et contient la conversation en cours
        context = self.session.get_context_string(session_id, last_n)
        if context:
            logger.debug(f"[MemoryManager] Contexte trouvé en RAM pour {session_id}")
            return context
        
        # ÉTAPE 2 : Si rien en RAM, chercher dans PersistentMemory (SQLite)
        # Cela arrive après un redémarrage du serveur
        logger.info(f"[MemoryManager] Chargement depuis SQLite pour {session_id}")
        history = self.persistent.get_session_history(session_id, limit=last_n)
        
        if history:
            # ÉTAPE 3 : Reconstruire le contexte sous forme de string
            lines = []
            for turn in history:
                lines.append(f"Q: {turn['query'][:200]}")
                lines.append(f"A: {turn['answer'][:300]}")
            context = "\n".join(lines)
            
            # ÉTAPE 4 : Restaurer dans SessionMemory pour les prochaines fois
            # Comme ça, les prochains appels seront plus rapides
            for turn in history:
                self.session.add_turn(session_id, turn['query'], turn['answer'])
            
            logger.info(f"[MemoryManager] Restauré {len(history)} tours depuis SQLite")
            return context
        
        # ÉTAPE 5 : Aucun historique trouvé
        logger.debug(f"[MemoryManager] Aucun historique pour {session_id}")
        return ""
    # ================================================================
    # FIN DE LA NOUVELLE PARTIE
    # ================================================================

    def get_stats(self) -> Dict:
        """Retourne les statistiques de la mémoire persistante"""
        return self.persistent.get_stats()


# Global singleton
memory_manager = MemoryManager()