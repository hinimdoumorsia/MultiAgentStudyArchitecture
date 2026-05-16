"""
PlanningAgent — Couche 2, Agent 1

Rôle : décompose la question académique en sous-tâches ordonnées,
estime la complexité et choisit la stratégie de résolution.
Communique avec l'orchestrateur (A2A : résultat → RouterDecision affiné).
"""

from typing import Dict, Any
import os  # ← AJOUTÉ pour os.getenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA  # ← CHANGÉ (remplace ChatGroq)
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.state import AcademicState

# ================================================================
# 🔥 NOUVEAU CODE AJOUTÉ - IMPORT DU MEMORY MANAGER 🔥
# ================================================================
from backend.memory.memory_manager import memory_manager
# ================================================================
# FIN DE LA NOUVELLE PARTIE
# ================================================================

SYSTEM_PROMPT = """Tu es un agent de planification pour un assistant académique.
Ta mission : analyser la question de l'utilisateur et produire un plan d'action structuré.

IMPORTANT : Si un historique de conversation t'est fourni (précédé par "HISTORIQUE RÉCENT"),
tu dois l'utiliser pour comprendre le contexte. Par exemple, si l'utilisateur a dit son nom
dans l'historique, tu dois t'en souvenir.

Réponds TOUJOURS en JSON avec ce format exact :
{
  "plan_steps": ["étape 1", "étape 2", ...],
  "complexity": "low|medium|high",
  "requires_retrieval": true|false,
  "requires_tools": true|false,
  "estimated_tokens": 500,
  "strategy": "description courte de la stratégie"
}
"""


class PlanningAgent(BaseAgent):
    name = "planning"
    description = (
        "Décompose la requête en sous-tâches ordonnées, estime la complexité "
        "et détermine quels autres agents sont nécessaires."
    )

    # 🔥 CHANGEMENT ICI : modèle NVIDIA
    def __init__(self, model: str = "meta/llama-3.1-70b-instruct"):
        self.llm = ChatNVIDIA(
            model=model,
            api_key=os.getenv("NVIDIA_API_KEY"),  # ← Clé NVIDIA dans .env
            max_tokens=1024,
            temperature=0.1
        )

    def process(self, state: AcademicState) -> Dict[str, Any]:
        
        # ================================================================
        # 🔥 NOUVEAU CODE AJOUTÉ - RÉCUPÉRATION DU CONTEXTE MÉMOIRE 🔥
        # ================================================================
        # Récupère l'historique de la conversation depuis la mémoire persistante
        contexte = memory_manager.get_session_context(state["session_id"])
        
        # Construit la question avec ou sans contexte
        if contexte:
            # Si un historique existe, on l'injecte dans la question
            question_avec_contexte = f"""[HISTORIQUE RÉCENT DE LA CONVERSATION]
{contexte}

[NOUVELLE QUESTION]
{state['user_query']}

IMPORTANT : Utilise l'historique ci-dessus pour comprendre le contexte.
Par exemple, si l'utilisateur a donné son nom dans l'historique, rappelle-toi de ce nom."""
            print(f"[PlanningAgent] ✅ Contexte chargé pour {state['session_id']}")
        else:
            # Pas d'historique, question normale
            question_avec_contexte = state['user_query']
            print(f"[PlanningAgent] ❌ Aucun contexte pour {state['session_id']}")
        # ================================================================
        # FIN DE LA NOUVELLE PARTIE
        # ================================================================
        
        # Utilise la question enrichie (avec contexte) au lieu de l'originale
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question académique : {question_avec_contexte}"),
        ]
        response = self.llm.invoke(messages)
        raw = response.content

        import json, re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        plan_data = json.loads(match.group()) if match else {}

        plan_text = (
            f"**Stratégie :** {plan_data.get('strategy', 'Analyse directe')}\n"
            f"**Complexité :** {plan_data.get('complexity', 'medium')}\n"
            f"**Étapes :**\n"
            + "\n".join(
                f"  {i+1}. {step}"
                for i, step in enumerate(plan_data.get("plan_steps", []))
            )
        )

        # 🔥 AJOUT DES TOKENS DANS LE RETURN
        tokens_data = {}
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            token_usage = response.response_metadata['token_usage']
            tokens_data = {
                "prompt_tokens": token_usage.get('prompt_tokens', 0),
                "completion_tokens": token_usage.get('completion_tokens', 0),
                "total_tokens": token_usage.get('total_tokens', 0)
            }
        else:
            # Fallback: estimation si les tokens ne sont pas disponibles
            tokens_data = {
                "prompt_tokens": len(str(messages)) // 4,
                "completion_tokens": len(raw) // 4,
                "total_tokens": (len(str(messages)) + len(raw)) // 4
            }

        return {
            "plan": plan_text,
            "router_decision": {
                **state.get("router_decision", {}),
                "estimated_complexity": plan_data.get("complexity", "medium"),
            },
            "tokens": tokens_data
        }

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        return {"query_length": len(state["user_query"])}