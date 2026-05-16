"""
SynthesisAgent — Couche 3

Rôle : agrège toutes les sorties des agents spécialisés en une réponse
finale argumentée, structurée et adaptée au niveau académique.
C'est le dernier nœud avant la réponse à l'utilisateur.
"""

from typing import Dict, Any
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.state import AcademicState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un agent de synthèse académique d'excellence.
Tu reçois les analyses de plusieurs agents spécialisés et tu dois produire
une réponse finale cohérente, bien structurée et argumentée.

Directives :
- Intègre les informations de tous les agents de façon harmonieuse
- Structure ta réponse avec des titres clairs (Markdown)
- Cite les sources quand disponibles
- Adapte le niveau au contexte académique (précis, rigoureux, pédagogique)
- Si le score de confiance est bas (<0.6), signale les incertitudes
- Termine par une conclusion synthétique
"""


class SynthesisAgent(BaseAgent):
    name = "synthesis"
    description = (
        "Agrège les sorties de tous les agents spécialisés en une réponse "
        "finale structurée, argumentée et adaptée au contexte académique."
    )

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.llm = ChatGroq(model=model, max_tokens=1024, temperature=0.1)

    def process(self, state: AcademicState) -> Dict[str, Any]:
        query = state["user_query"]
        plan = state.get("plan", "")
        rag_output = state.get("retrieved_docs", "")
        tool_output = state.get("tool_results", "")
        
        # CORRECTION: Vérifier que verification_report n'est pas None
        verification = state.get("verification_report")
        if verification is None:
            verification = {}
            logger.warning("[SynthesisAgent] verification_report is None, using empty dict")

        confidence = verification.get("confidence_score", 0.8)
        recommendation = verification.get("recommendation", "PROCEED")
        notes = verification.get("verification_notes", "")

        # Build context for synthesis
        sections = [f"**Question :** {query}\n"]
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
                "⚠️ **Avertissement :** Score de confiance bas — "
                f"{confidence:.0%}. Certaines informations peuvent être incomplètes.\n"
            )

        context = "\n---\n".join(sections)

        prompt = (
            f"Synthétise les informations suivantes en une réponse académique complète.\n\n"
            f"{context}\n\n"
            f"Produis maintenant la réponse finale structurée."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        if recommendation == "FALLBACK":
            # Simplified response if verification recommends fallback
            messages[1] = HumanMessage(
                content=f"Réponds directement à cette question académique : {query}"
            )

        try:
            response = self.llm.invoke(messages)
            
            # 🔥 AJOUT DES TOKENS
            tokens_data = {}
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                token_usage = response.response_metadata['token_usage']
                tokens_data = {
                    "prompt_tokens": token_usage.get('prompt_tokens', 0),
                    "completion_tokens": token_usage.get('completion_tokens', 0),
                    "total_tokens": token_usage.get('total_tokens', 0)
                }
            else:
                # Fallback: estimation
                tokens_data = {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(response.content) // 4,
                    "total_tokens": (len(prompt) + len(response.content)) // 4
                }
            
            return {
                "final_answer": response.content,
                "tokens": tokens_data  # ← AJOUT ICI
            }
        except Exception as e:
            logger.error(f"[SynthesisAgent] LLM error: {e}")
            # Fallback response
            return {
                "final_answer": f"Je n'ai pas pu générer une réponse complète. Erreur: {str(e)[:100]}",
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # ← AJOUT
            }

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        # CORRECTION: Vérifier que verification_report n'est pas None
        verification = state.get("verification_report")
        if verification is None:
            verification = {}
        return {
            "confidence_score": verification.get("confidence_score", 0.8),
            "recommendation": verification.get("recommendation", "PROCEED"),
        }