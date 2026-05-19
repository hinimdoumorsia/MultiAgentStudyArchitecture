"""
PlanningAgent — Couche 2, Agent 1

Rôle : décompose la question académique en sous-tâches ordonnées,
estime la complexité et choisit la stratégie de résolution.
"""

from typing import Dict, Any
import os
import asyncio
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.state import AcademicState
from backend.memory.memory_manager import memory_manager

SYSTEM_PROMPT = """Tu es un agent de planification pour un assistant académique.
Ta mission : analyser la question de l'utilisateur et produire un plan d'action structuré.

IMPORTANT : Si un historique de conversation t'est fourni (précédé par "HISTORIQUE RÉCENT"),
tu dois l'utiliser pour comprendre le contexte.

Réponds TOUJOURS en JSON avec ce format exact :
{
  "plan_steps": ["étape 1", "étape 2", ...],
  "complexity": "low|medium|high",
  "requires_retrieval": true|false,
  "requires_tools": true|false,
  "estimated_tokens": 500,
  "strategy": "description courte"
}

Pour les questions de CODE, sois plus concis. Exemple:
{
  "plan_steps": ["Analyser la demande", "Générer le code", "Expliquer la solution"],
  "complexity": "medium",
  "requires_retrieval": false,
  "requires_tools": true,
  "estimated_tokens": 300,
  "strategy": "Génération de code Python avec explication"
}
"""

class PlanningAgent(BaseAgent):
    name = "planning"
    description = "Décompose la requête en sous-tâches ordonnées"

    def __init__(self, model: str = "meta/llama-3.1-8b-instruct"):
        # 🔥 Utilisation d'un modèle plus léger pour la planification
        # 🔥 SUPPRESSION du paramètre timeout (non supporté par l'API)
        self.llm = ChatNVIDIA(
            model=model,
            api_key=os.getenv("NVIDIA_API_KEY"),
            max_tokens=512,
            temperature=0.1
            # timeout SUPPRIMÉ - cause l'erreur "Unsupported parameter(s): timeout"
        )

    def process(self, state: AcademicState) -> Dict[str, Any]:
        
        contexte = memory_manager.get_session_context(state["session_id"])
        
        if contexte:
            question_avec_contexte = f"""[HISTORIQUE RÉCENT]
{contexte}

[QUESTION]
{state['user_query']}"""
            print(f"[PlanningAgent] ✅ Contexte chargé")
        else:
            question_avec_contexte = state['user_query']
            print(f"[PlanningAgent] ❌ Aucun contexte")
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question : {question_avec_contexte}"),
        ]
        
        try:
            response = self.llm.invoke(messages)
            raw = response.content
        except Exception as e:
            print(f"[PlanningAgent] ⚠️ Erreur LLM: {e}")
            return self._fallback_plan(state)
        
        import json, re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        plan_data = json.loads(match.group()) if match else {}

        if not plan_data.get("plan_steps"):
            return self._fallback_plan(state)
        
        plan_text = (
            f"**Stratégie :** {plan_data.get('strategy', 'Analyse directe')}\n"
            f"**Complexité :** {plan_data.get('complexity', 'medium')}\n"
            f"**Étapes :**\n"
            + "\n".join(
                f"  {i+1}. {step}"
                for i, step in enumerate(plan_data.get("plan_steps", []))
            )
        )

        tokens_data = {}
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            token_usage = response.response_metadata['token_usage']
            tokens_data = {
                "prompt_tokens": token_usage.get('prompt_tokens', 0),
                "completion_tokens": token_usage.get('completion_tokens', 0),
                "total_tokens": token_usage.get('total_tokens', 0)
            }
        else:
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
    
    def _fallback_plan(self, state: AcademicState) -> Dict[str, Any]:
        """Plan de secours en cas d'erreur ou timeout"""
        query = state["user_query"]
        
        if any(word in query.lower() for word in ["code", "python", "implémenter", "programme"]):
            plan_text = """**Stratégie :** Génération de code
**Complexité :** medium
**Étapes :**
  1. Analyser la demande de code
  2. Générer une solution
  3. Expliquer le code"""
            complexity = "medium"
        elif any(word in query.lower() for word in ["somme", "calcul", "math", "équation"]):
            plan_text = """**Stratégie :** Calcul direct
**Complexité :** low
**Étapes :**
  1. Effectuer le calcul
  2. Fournir le résultat"""
            complexity = "low"
        else:
            plan_text = """**Stratégie :** Analyse et réponse
**Complexité :** medium
**Étapes :**
  1. Comprendre la question
  2. Rechercher l'information
  3. Synthétiser la réponse"""
            complexity = "medium"
        
        return {
            "plan": plan_text,
            "router_decision": {
                **state.get("router_decision", {}),
                "estimated_complexity": complexity,
            },
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        return {"query_length": len(state["user_query"])}