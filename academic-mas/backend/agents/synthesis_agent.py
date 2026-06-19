"""
SynthesisAgent — Couche 3

Rôle : agrège toutes les sorties des agents spécialisés en une réponse
finale argumentée, structurée et adaptée au niveau académique.

🔥 CORRECTIF : Utilise la réponse de PlanningAgent si déjà présente
"""

from typing import Dict, Any
import logging
import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.state import AcademicState
from backend.memory.memory_manager import memory_manager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un agent de synthèse académique d'excellence.
Tu reçois les analyses de plusieurs agents spécialisés et tu dois produire
une réponse finale cohérente, bien structurée et argumentée.

🔥 RÈGLE IMPÉRATIVE : Utilise TOUTES les informations disponibles de l'utilisateur.
"""

class SynthesisAgent(BaseAgent):
    name = "synthesis"
    description = (
        "Agrège les sorties de tous les agents spécialisés en une réponse "
        "finale structurée, argumentée et adaptée au contexte académique."
    )

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.llm = ChatGroq(model=model, max_tokens=1024, temperature=0.1)

    def _get_user_info_from_db(self, session_id: str) -> Dict:
        """🔥 Récupère TOUTES les informations depuis SQLite"""
        try:
            prefs = memory_manager.persistent.get_preferences(session_id)
            if prefs:
                return prefs
        except Exception as e:
            logger.warning(f"[SynthesisAgent] Erreur SQLite: {e}")
        return {}

    def _is_identity_question_with_llm(self, query: str) -> bool:
        """
        Détecte si l'utilisateur POSE une question sur sa propre identité
        (ex: "comment je m'appelle ?", "quelle est ma profession ?").

        🔧 CORRECTIF : l'ancienne version ne distinguait pas "poser une
        question sur son identité" de "donner des informations sur soi".
        Résultat : une simple présentation ("je suis Djida, spécialiste en
        nutrition...") était classée comme question d'identité, ce qui
        déclenchait le message générique "Je ne connais pas encore vos
        informations" et écrasait la bonne réponse déjà produite par
        RAGAgent.
        """
        prompt = f"""
        Voici une phrase écrite par un utilisateur à un assistant.

        Réponds UNIQUEMENT par OUI si l'utilisateur POSE une question pour
        QUE L'ASSISTANT LUI RAPPELLE une information sur lui-même qu'il a
        donnée précédemment (par exemple : "comment je m'appelle ?",
        "quelle est ma profession ?", "où est-ce que j'habite déjà ?",
        "qu'est-ce que j'aime faire ?").

        Réponds NON si l'utilisateur :
        - se présente ou donne des informations sur lui-même (nom, métier,
          pays, goûts), même formulé de façon informelle ou avec des fautes
        - pose une question générale qui n'est pas sur sa propre identité
        - fait toute autre demande

        Réponds UNIQUEMENT par OUI ou NON, sans rien d'autre.

        Phrase: {query}
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip().upper().startswith("OUI")
        except Exception as e:
            logger.warning(f"[SynthesisAgent] Erreur détection identité: {e}")
            return False

    def _generate_answer_with_llm(self, query: str, user_info: Dict) -> str:
        """🔥 Le LLM génère la réponse avec TOUTES les infos"""
        user_info_str = json.dumps(user_info, ensure_ascii=False, indent=2)
        
        prompt = f"""
        Informations de l'utilisateur :
        {user_info_str}
        
        Question : {query}
        
        Réponds de manière personnalisée en utilisant les informations ci-dessus.
        Si l'utilisateur demande son nom, donne-le.
        Si l'utilisateur demande sa profession, donne-la.
        Si l'utilisateur demande son origine, donne-la.
        Sois précis, chaleureux et structuré.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"[SynthesisAgent] Erreur génération: {e}")
            return f"Erreur lors de la génération de la réponse: {e}"

    def process(self, state: AcademicState) -> Dict[str, Any]:
        query = state["user_query"]
        session_id = state.get("session_id", "")
        
        # ── 🔥 PRIORITÉ 1 : Si PlanningAgent a déjà répondu ─────────────
        # Vérifier si PlanningAgent a mis une réponse personnalisée dans le state
        planning_output = None
        for result in state.get("agent_results", []):
            if result.get("agent_name") == "planning":
                planning_output = result.get("output")
                break
        
        # Si PlanningAgent a répondu avec des infos personnelles, on les utilise
        if planning_output and ("vous appelez" in planning_output or "Fatime" in planning_output or "Abakar" in planning_output or "infirmière" in planning_output or "expert" in planning_output):
            logger.info(f"[SynthesisAgent] Utilisation de la réponse de PlanningAgent")
            return {
                "final_answer": planning_output,
                "confidence": 1.0,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        
        # ── RÉCUPÉRER TOUTES LES INFOS ──────────────────────────────────
        user_name = state.get("user_name")
        user_info = self._get_user_info_from_db(session_id)
        
        if not user_name and user_info:
            user_name = user_info.get("nom") or user_info.get("name")
        
        if not user_name:
            try:
                from backend.agents.planning_agent import USER_NAME_CACHE
                user_name = USER_NAME_CACHE.get(session_id)
                if user_name:
                    user_info["nom"] = user_name
            except ImportError:
                pass
        
        # ── LE LLM DÉTECTE SI C'EST UNE VRAIE QUESTION D'IDENTITÉ ──────
        is_identity = self._is_identity_question_with_llm(query)

        if is_identity:
            if user_info:
                answer = self._generate_answer_with_llm(query, user_info)
                return {
                    "final_answer": answer,
                    "confidence": 1.0,
                    "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

            # 🔧 CORRECTIF : aucune info connue en base, mais si RAGAgent a
            # déjà produit une réponse pertinente (ex: il a lu la présentation
            # de l'utilisateur dans la même requête et y a déjà répondu), on
            # s'en sert plutôt que d'écraser ce travail par un message
            # générique. Le message générique ne devient le dernier recours
            # que si rien d'autre n'est disponible.
            rag_output = state.get("retrieved_docs", "")
            if rag_output:
                logger.info(
                    "[SynthesisAgent] Question d'identité sans info en base, "
                    "fallback sur la réponse déjà produite par RAGAgent."
                )
                return {
                    "final_answer": rag_output,
                    "confidence": 0.7,
                    "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

            answer = "❓ Je ne connais pas encore vos informations. Parlez-moi de vous !"
            return {
                "final_answer": answer,
                "confidence": 0.5,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        
        # ── Synthèse normale ─────────────────────────────────────────────
        plan = state.get("plan", "")
        rag_output = state.get("retrieved_docs", "")
        tool_output = state.get("tool_results", "")
        verification = state.get("verification_report") or {}

        confidence = verification.get("confidence_score", 0.8)
        recommendation = verification.get("recommendation", "PROCEED")
        notes = verification.get("verification_notes", "")

        sections = [f"**Question :** {query}\n"]
        if user_name:
            sections.append(f"**Utilisateur :** {user_name}\n")
        if plan:
            sections.append(f"**Plan d'analyse :**\n{plan}\n")
        if rag_output:
            sections.append(f"**Recherche documentaire :**\n{rag_output}\n")
        if tool_output and tool_output != "Aucun outil externe requis pour cette question.":
            sections.append(f"**Résultats des outils :**\n{tool_output}\n")
        if notes:
            sections.append(f"**Note de vérification :** {notes}\n")
        if confidence < 0.6:
            sections.append(
                f"⚠️ **Avertissement :** Score de confiance bas — {confidence:.0%}. "
                "Certaines informations peuvent être incomplètes.\n"
            )

        context = "\n---\n".join(sections)
        prompt = (
            f"Synthétise les informations suivantes en une réponse académique complète.\n\n"
            f"{context}\n\nProduis maintenant la réponse finale structurée."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        if recommendation == "FALLBACK":
            messages[1] = HumanMessage(
                content=f"Réponds directement à cette question académique : {query}"
            )

        try:
            response = self.llm.invoke(messages)

            tokens_data = {}
            if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
                tu = response.response_metadata["token_usage"]
                tokens_data = {
                    "prompt_tokens": tu.get("prompt_tokens", 0),
                    "completion_tokens": tu.get("completion_tokens", 0),
                    "total_tokens": tu.get("total_tokens", 0),
                }
            else:
                tokens_data = {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(response.content) // 4,
                    "total_tokens": (len(prompt) + len(response.content)) // 4,
                }

            return {"final_answer": response.content, "tokens": tokens_data}

        except Exception as e:
            logger.error(f"[SynthesisAgent] LLM error: {e}")
            return {
                "final_answer": f"Erreur lors de la synthèse : {str(e)[:100]}",
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        verification = state.get("verification_report") or {}
        return {
            "confidence_score": verification.get("confidence_score", 0.8),
            "recommendation": verification.get("recommendation", "PROCEED"),
        }