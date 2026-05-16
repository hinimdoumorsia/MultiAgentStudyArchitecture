"""
VerificationAgent — Couche 2, Agent 4

Rôle : évalue la cohérence, détecte les hallucinations potentielles,
et produit un score de confiance pour la synthèse finale.
Protocole A2A : lit les sorties des agents précédents via l'état partagé.
"""

from typing import Dict, Any
import json
import re
import logging
import time

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.state import AcademicState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un agent de vérification académique rigoureux.
Tu reçois : la question originale, le plan, les documents RAG, et les résultats d'outils.
Ton rôle : vérifier la cohérence, détecter les incohérences et produire un rapport.

Réponds UNIQUEMENT en JSON :
{
  "confidence_score": 0.0-1.0,
  "consistency_check": "Cohérent|Incohérent|Partiel",
  "potential_hallucinations": ["liste d'affirmations douteuses"],
  "missing_information": ["ce qui manque"],
  "quality_score": 0.0-1.0,
  "recommendation": "PROCEED|RETRY|FALLBACK",
  "verification_notes": "commentaire court"
}
"""


class VerificationAgent(BaseAgent):
    name = "verification"
    description = (
        "Vérifie la cohérence des sorties des autres agents, détecte les "
        "hallucinations potentielles et produit un score de confiance global."
    )

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.llm = ChatGroq(model=model, max_tokens=1024, temperature=0.1)

    def process(self, state: AcademicState) -> Dict[str, Any]:
        # Attendre 2 secondes pour éviter rate limit
        time.sleep(3)
        
        query = state["user_query"]
        
        # 🔥 CORRECTION : Gérer les valeurs None
        plan = state.get("plan")
        rag_output = state.get("retrieved_docs")
        tool_output = state.get("tool_results")
        
        # Convertir None en chaînes vides
        plan_text = plan if plan is not None else "Aucun plan disponible"
        rag_text = rag_output if rag_output is not None else "Aucun document récupéré"
        tool_text = tool_output if tool_output is not None else "Aucun outil utilisé"

        prompt = f"""Question originale : {query}

Plan d'action :
{plan_text[:500]}

Résultats RAG :
{rag_text[:800]}

Résultats des outils :
{tool_text[:500] if tool_text else "Aucun outil utilisé"}

Vérifie la cohérence de ces informations et produis ton rapport JSON."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
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
                
        except Exception as e:
            logger.error(f"[VerificationAgent] API error: {e}")
            # Retourner un rapport par défaut en cas d'erreur
            report = {
                "confidence_score": 0.5,
                "consistency_check": "Partiel",
                "potential_hallucinations": [],
                "missing_information": [],
                "quality_score": 0.5,
                "recommendation": "PROCEED",
                "verification_notes": f"Erreur API: {str(e)[:100]}",
            }
            return {
                "verification_report": report,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        
        raw = response.content

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            report = json.loads(match.group())
        else:
            report = {
                "confidence_score": 0.7,
                "consistency_check": "Partiel",
                "potential_hallucinations": [],
                "missing_information": [],
                "quality_score": 0.7,
                "recommendation": "PROCEED",
                "verification_notes": "Vérification automatique (parsing échoué)",
            }

        logger.info(
            f"[VerificationAgent] Score: {report.get('confidence_score', '?')} "
            f"| Rec: {report.get('recommendation', '?')} "
            f"| Tokens: {tokens_data.get('total_tokens', 0)}"
        )
        
        return {
            "verification_report": report,
            "tokens": tokens_data
        }

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        report = state.get("verification_report")
        if report is None:
            report = {}
        return {
            "confidence_score": report.get("confidence_score", 0),
            "recommendation": report.get("recommendation", "UNKNOWN"),
        }