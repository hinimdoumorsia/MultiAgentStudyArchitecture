"""
MemoryManager — Gestion de la mémoire à deux niveaux

- Mémoire de session : contexte de la conversation en cours (in-memory)
- Mémoire persistante : historique inter-sessions (SQLite)
- 🔥 Extraction et stockage des préférences utilisateur (nom inclus)
"""

import sqlite3
import json
import os
import time
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionMemory:
    """In-memory session store. Cleared on restart."""

    def __init__(self):
        self._sessions: Dict[str, List[Dict]] = {}
        self._preferences: Dict[str, Dict] = {}

    def add_turn(self, session_id: str, query: str, answer: str, metadata: Dict = None):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({
            "timestamp": time.time(),
            "query": query,
            "answer": answer,
            "metadata": metadata or {},
        })
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
        self._preferences.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def set_preferences(self, session_id: str, preferences: Dict):
        if session_id not in self._preferences:
            self._preferences[session_id] = {}
        self._preferences[session_id].update(preferences)
        logger.info(f"[SessionMemory] Préférences mises à jour pour {session_id}: {preferences}")

    def get_preferences(self, session_id: str) -> Dict:
        return self._preferences.get(session_id, {})

    def get_preference_context(self, session_id: str) -> str:
        prefs = self.get_preferences(session_id)
        if not prefs:
            return ""
        lines = ["Préférences de l'utilisateur :"]
        for key, value in prefs.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)


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
                CREATE TABLE IF NOT EXISTS preferences (
                    session_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context (
                    session_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    PRIMARY KEY (session_id, topic)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_preferences ON preferences(session_id)")
            conn.commit()
        logger.info(f"[PersistentMemory] DB ready at {self.db_path}")

    def save_conversation(self, session_id, run_id, query, answer,
                          agents_used, confidence, latency_ms, metadata=None):
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

    def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """🔥 Récupère l'historique complet d'une session (jusqu'à 50 messages)"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT query, answer, agents_used, confidence, latency_ms, timestamp
                FROM conversations WHERE session_id = ?
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

    def save_preferences(self, session_id: str, preferences: Dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO preferences (session_id, preferences, updated_at)
                VALUES (?, ?, ?)
            """, (session_id, json.dumps(preferences), datetime.now().isoformat()))
            conn.commit()
        logger.info(f"[PersistentMemory] Préférences sauvegardées pour {session_id}")

    def get_preferences(self, session_id: str) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT preferences FROM preferences WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            if row:
                return json.loads(row[0])
        return {}

    def save_context(self, session_id: str, topic: str, value: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO context (session_id, topic, value, timestamp)
                VALUES (?, ?, ?, ?)
            """, (session_id, topic, value, datetime.now().isoformat()))
            conn.commit()
        logger.info(f"[PersistentMemory] Contexte sauvegardé: {topic}={value}")

    def get_context(self, session_id: str, topic: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM context WHERE session_id = ? AND topic = ?",
                (session_id, topic)
            ).fetchone()
            if row:
                return row[0]
        return None

    def search_history(self, session_id: str, keywords: List[str], limit: int = 5) -> List[Dict]:
        if not keywords:
            return []
        conditions = []
        params = [session_id]
        for kw in keywords:
            conditions.append("(query LIKE ? OR answer LIKE ?)")
            params.extend([f"%{kw}%", f"%{kw}%"])
        query = f"""
            SELECT query, answer, timestamp FROM conversations
            WHERE session_id = ? AND ({' OR '.join(conditions)})
            ORDER BY id DESC LIMIT ?
        """
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [{"query": r[0], "answer": r[1], "timestamp": r[2]} for r in rows]

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

        # 🔥 Patterns pour détecter le NOM de l'utilisateur
        # CORRECTION : s'arrête avant "et toi", ",", ".", "?", "!"
        self.name_patterns = [
            r"je\s+(?:me\s+nomme|m'appelle|suis)\s+([A-Za-zÀ-ÿ\s\-]{2,50})(?:\s+et\s+toi|,|\.|\?|!|$)",
            r"mon\s+nom\s+(?:est|c'est)\s+([A-Za-zÀ-ÿ\s\-]{2,50})(?:\s+et\s+toi|,|\.|\?|!|$)",
            r"appelle[z\s]+moi\s+([A-Za-zÀ-ÿ\s\-]{2,50})(?:\s+et\s+toi|,|\.|\?|!|$)",
            r"my\s+name\s+is\s+([A-Za-zÀ-ÿ\s\-]{2,50})(?:\s+et\s+toi|,|\.|\?|!|$)",
            r"i\s+am\s+([A-Za-zÀ-ÿ\s\-]{2,50})(?:\s+et\s+toi|,|\.|\?|!|$)",
        ]

        # Patterns généraux pour les autres préférences
        self.preference_patterns = [
            (r"j'aime\s+([^.!?\n]{3,80})", "centres_d_interet"),
            (r"je préfère\s+([^.!?\n]{3,80})", "preference"),
            (r"je déteste\s+([^.!?\n]{3,80})", "a_eviter"),
            (r"je n'aime pas\s+([^.!?\n]{3,80})", "a_eviter"),
            (r"mon domaine est\s+([^.!?\n]{3,80})", "domaine"),
            (r"je travaille (?:sur|dans|en)\s+([^.!?\n]{3,80})", "domaine"),
            (r"mon objectif est\s+([^.!?\n]{3,80})", "objectif"),
        ]

    # ─────────────────────────────────────────────
    # 🔥 EXTRACTION DU NOM — méthode publique
    # ─────────────────────────────────────────────
    def extract_user_name(self, text: str) -> Optional[str]:
        """
        Tente d'extraire un nom depuis un texte.
        Retourne le nom nettoyé ou None si rien trouvé.
        """
        for pattern in self.name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip().rstrip(".,!?")
                # Filtrer les faux positifs trop courts ou trop longs
                if 2 <= len(name) <= 60:
                    return name
        return None

    # ─────────────────────────────────────────────
    # 🔥 CORRECTION : get_user_name améliore
    # ─────────────────────────────────────────────
    def get_user_name(self, session_id: str) -> Optional[str]:
        """
        Récupère le nom de l'utilisateur :
        1. Depuis la mémoire session (RAM)
        2. Depuis SQLite (persistant)
        
        🔥 CORRECTION : Cherche "nom" ou "name" dans toutes les clés
        """
        # RAM d'abord
        prefs = self.session.get_preferences(session_id)
        if prefs:
            for key, value in prefs.items():
                if "nom" in key.lower() or "name" in key.lower():
                    return value
            return list(prefs.values())[0] if prefs else None

        # SQLite ensuite
        prefs_db = self.persistent.get_preferences(session_id)
        if prefs_db:
            for key, value in prefs_db.items():
                if "nom" in key.lower() or "name" in key.lower():
                    self.session.set_preferences(session_id, {"nom": value})
                    return value
            value = list(prefs_db.values())[0] if prefs_db else None
            if value:
                self.session.set_preferences(session_id, {"nom": value})
                return value

        return None

    def record(self, session_id: str, run_id: str, query: str, answer: str,
               agents_used: List[str], confidence: float, latency_ms: float):
        """Enregistre une conversation + extrait préférences et nom."""
        self.session.add_turn(session_id, query, answer)
        self.persistent.save_conversation(
            session_id, run_id, query, answer,
            agents_used, confidence, latency_ms,
        )
        self._extract_and_store_preferences(session_id, query)

    def _extract_and_store_preferences(self, session_id: str, query: str):
        """
        🔥 Extrait le NOM et les autres préférences depuis la requête.
        Stocke immédiatement en RAM + SQLite.
        """
        preferences = {}

        # ── 1. Extraction du NOM (prioritaire) ──────────────────────────
        name = self.extract_user_name(query)
        if name:
            preferences["nom"] = name
            logger.info(f"[MemoryManager] Nom détecté pour {session_id}: '{name}'")

        # ── 2. Autres préférences ────────────────────────────────────────
        for pattern, key in self.preference_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value = match.group(1).strip().rstrip(".,!?")
                if value:
                    preferences[key] = value

        # ── 3. Stocker si quelque chose trouvé ──────────────────────────
        if preferences:
            self.session.set_preferences(session_id, preferences)
            existing = self.persistent.get_preferences(session_id)
            existing.update(preferences)
            self.persistent.save_preferences(session_id, existing)
            for key, value in preferences.items():
                self.persistent.save_context(session_id, key, value)
            logger.info(f"[MemoryManager] Préférences stockées pour {session_id}: {preferences}")

    def get_session_context(self, session_id: str, last_n: int = 3) -> str:
        """
        Récupère le contexte de session.
        Charge depuis SQLite si la RAM est vide (redémarrage serveur).
        """
        context = self.session.get_context_string(session_id, last_n)

        if not context:
            logger.info(f"[MemoryManager] Chargement depuis SQLite pour {session_id}")
            history = self.persistent.get_session_history(session_id, limit=last_n)
            if history:
                lines = []
                for turn in history:
                    lines.append(f"Q: {turn['query'][:200]}")
                    lines.append(f"A: {turn['answer'][:300]}")
                context = "\n".join(lines)
                for turn in history:
                    self.session.add_turn(session_id, turn["query"], turn["answer"])

        # Toujours charger les préférences depuis SQLite si RAM vide
        if not self.session.get_preferences(session_id):
            prefs = self.persistent.get_preferences(session_id)
            if prefs:
                self.session.set_preferences(session_id, prefs)

        prefs_context = self.session.get_preference_context(session_id)
        if prefs_context:
            context = f"{context}\n\n{prefs_context}" if context else prefs_context

        return context

    def get_preferences(self, session_id: str) -> Dict:
        prefs = self.session.get_preferences(session_id)
        if prefs:
            return prefs
        return self.persistent.get_preferences(session_id)

    def get_context_for_query(self, session_id: str, query: str) -> str:
        stopwords = {
            "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
            "le", "la", "les", "un", "une", "des", "pour", "par", "avec",
            "sans", "dans", "sur", "sous", "entre", "et", "ou", "donc",
            "or", "ni", "car", "mais", "que", "qui", "quoi", "dont",
        }
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stopwords]

        if not keywords:
            return self.get_session_context(session_id)

        results = self.persistent.search_history(session_id, keywords, limit=3)
        if results:
            lines = ["Contexte pertinent trouvé :"]
            for r in results:
                lines.append(f"- {r['query']} → {r['answer'][:200]}")
            context = "\n".join(lines)
            prefs = self.get_preferences(session_id)
            if prefs:
                prefs_lines = ["Préférences connues :"]
                for key, value in prefs.items():
                    prefs_lines.append(f"- {key}: {value}")
                context = f"{context}\n\n" + "\n".join(prefs_lines)
            return context

        return self.get_session_context(session_id)

    def get_stats(self) -> Dict:
        return self.persistent.get_stats()


# Global singleton
memory_manager = MemoryManager()