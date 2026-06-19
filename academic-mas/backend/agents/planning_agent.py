"""
PlanningAgent — Couche 2, Agent 1

Rôle : décompose la question académique en sous-tâches ordonnées,
estime la complexité et choisit la stratégie de résolution.
"""

from typing import Dict, Any, List
import os
import asyncio
import re
import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.state import AcademicState
from backend.memory.memory_manager import memory_manager

# ── Cache de session ultra-rapide ─────────────────────────────────────
USER_NAME_CACHE = {}

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
        self.llm = ChatNVIDIA(
            model=model,
            api_key=os.getenv("NVIDIA_API_KEY"),
            max_tokens=512,
            temperature=0.1
        )

    def _extract_all_user_info_with_llm(self, query: str) -> Dict[str, Any]:
        """
        🔥 Le LLM extrait TOUTES les informations personnelles de l'utilisateur.
        """
        prompt = f"""
        Extrais toutes les informations personnelles de l'utilisateur dans cette phrase.
        
        Retourne UNIQUEMENT du JSON avec ces champs :
        {{
            "nom": "prénom nom" ou null,
            "profession": "profession" ou null,
            "expertise": "domaine d'expertise" ou null,
            "preferences": "ce qu'il aime" ou null
        }}
        
        Phrase: {query}
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            # Nettoyer si besoin (enlever les balises markdown)
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            return json.loads(content)
        except Exception as e:
            print(f"[PlanningAgent] Erreur extraction infos: {e}")
            return {}

    def _answer_with_full_history(self, query: str, history: List[Dict], user_name: str = None) -> str:
        """
        🔥 Répond en utilisant TOUT l'historique de la conversation.
        """
        if not history and not user_name:
            return "❓ Je ne connais pas encore vos informations. Parlez-moi de vous !"
        
        # Construire l'historique
        history_text = ""
        if history:
            history_text = "\n".join([
                f"Q: {h['query']}\nA: {h['answer']}" 
                for h in history[-10:]  # Derniers 10 échanges
            ])
        
        prompt = f"""
        Tu es un assistant qui répond avec des informations personnelles.
        
        {f"L'utilisateur s'appelle: {user_name}" if user_name else ""}
        
        Historique de la conversation :
        {history_text if history_text else "Aucun historique"}
        
        Question actuelle : {query}
        
        Réponds de manière personnalisée en utilisant les informations de l'historique.
        Si l'utilisateur demande son nom, donne-le.
        Si l'utilisateur demande sa profession, donne-la.
        Sois précis et chaleureux.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Erreur lors de la réponse avec historique: {e}"

    def process(self, state: AcademicState) -> Dict[str, Any]:
        query = state["user_query"]
        session_id = state.get("session_id", "default")
        
        # ── RÉCUPÉRATION DE L'HISTORIQUE COMPLET ──────────────────────
        history = memory_manager.persistent.get_session_history(session_id, limit=50)
        
        # ── RÉCUPÉRATION DES PRÉFÉRENCES EXISTANTES ──────────────────
        existing_prefs = memory_manager.persistent.get_preferences(session_id)
        user_name = existing_prefs.get("nom") or USER_NAME_CACHE.get(session_id)
        
        if not user_name:
            user_name = memory_manager.get_user_name(session_id)
            if user_name:
                USER_NAME_CACHE[session_id] = user_name
        
        # ── LE LLM EXTRAIT TOUTES LES INFOS DE LA REQUÊTE ─────────────
        new_user_info = self._extract_all_user_info_with_llm(query)
        
        # ── STOCKER TOUT EN SQLITE (FUSION) ──────────────────────────
        if new_user_info:
            # Fusionner avec les existantes
            existing_prefs.update(new_user_info)
            memory_manager.persistent.save_preferences(session_id, existing_prefs)
            
            # Mettre à jour le cache
            if new_user_info.get("nom"):
                user_name = new_user_info["nom"]
                USER_NAME_CACHE[session_id] = user_name
                print(f"[PlanningAgent] ✅ NOM ENREGISTRÉ: {user_name}")
            
            print(f"[PlanningAgent] ✅ Infos enregistrées: {new_user_info}")
        
        # ── DÉTECTION DES QUESTIONS D'IDENTITÉ ────────────────────────
        identity_keywords = [
            "nom", "m'appelle", "me nomme", "qui suis", "profession",
            "expert", "que fais", "présenter", "qui je suis",
            "mon métier", "mon travail", "je fais quoi"
        ]
        
        is_identity_question = any(kw in query.lower() for kw in identity_keywords)
        
        if is_identity_question:
            # Utiliser TOUT l'historique pour répondre
            final_answer = self._answer_with_full_history(query, history, user_name)
            
            return {
                "user_name": user_name,
                "plan": f"**Stratégie :** Réponse basée sur l'historique complet\n**Étapes :**\n  1. Charger l'historique\n  2. Répondre avec les informations",
                "final_answer": final_answer,
                "router_decision": {**state.get("router_decision", {}), "estimated_complexity": "low"},
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        
        # ── DÉTECTION DES DÉCLARATIONS DE NOM ──────────────────────────
        name_patterns = [
            r"(?:je m'appelle|je me nomme|mon nom est|appelez-moi|moi c'est|je suis|je repond a ce nom)\s+([A-Za-zÀ-ÿ\s\-]{2,50})",
        ]
        
        is_name_declaration = False
        for pattern in name_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                is_name_declaration = True
                break
        
        if is_name_declaration:
            extracted_name = new_user_info.get("nom") or self._extract_all_user_info_with_llm(query).get("nom")
            if extracted_name:
                user_name = extracted_name
                USER_NAME_CACHE[session_id] = user_name
                existing_prefs["nom"] = user_name
                memory_manager.persistent.save_preferences(session_id, existing_prefs)
                print(f"[PlanningAgent] ✅ NOM ENREGISTRÉ: {user_name}")
                return {
                    "user_name": user_name,
                    "plan": f"**Stratégie :** Enregistrement du nom\n**Étapes :**\n  1. Nom '{user_name}' enregistré",
                    "final_answer": f"👋 Enchanté {user_name} ! Je suis votre assistant.",
                    "router_decision": {**state.get("router_decision", {}), "estimated_complexity": "low"},
                    "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
        
        # ── SUITE DU TRAITEMENT NORMAL ──────────────────────────────────
        contexte = memory_manager.get_session_context(session_id)
        
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
            "user_name": user_name,
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
        session_id = state.get("session_id", "default")
        user_name = USER_NAME_CACHE.get(session_id)
        
        if any(word in query.lower() for word in ["code", "python", "implémenter", "programme"]):
            plan_text = """**Stratégie :** Génération de code\n**Complexité :** medium\n**Étapes :**\n  1. Analyser la demande\n  2. Générer la solution"""
            complexity = "medium"
        elif any(word in query.lower() for word in ["somme", "calcul", "math", "équation"]):
            plan_text = """**Stratégie :** Calcul direct\n**Complexité :** low\n**Étapes :**\n  1. Effectuer le calcul\n  2. Fournir le résultat"""
            complexity = "low"
        else:
            plan_text = """**Stratégie :** Analyse et réponse\n**Complexité :** medium\n**Étapes :**\n  1. Comprendre la question\n  2. Rechercher l'information\n  3. Synthétiser la réponse"""
            complexity = "medium"
        
        return {
            "user_name": user_name,
            "plan": plan_text,
            "router_decision": {
                **state.get("router_decision", {}),
                "estimated_complexity": complexity,
            },
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        return {"query_length": len(state["user_query"])}