"""
DynamicRouter — Routeur intelligent basé sur des signaux multiples

Signaux de routage :
- Type de requête (calcul, recherche, explication, code…)
- Score de confiance estimé
- Charge contextuelle (longueur du contexte session)
- Complexité détectée par l'agent de planification
"""

import re
from typing import List, Dict
import logging

from backend.state import AcademicState, RouterDecision
from backend.agents.registry import registry

logger = logging.getLogger(__name__)

# ── Routing rules (keyword-based signals) ────────────────────────────

MATH_PATTERNS = re.compile(
    r'\b(calcul|calculer|mathémat|équation|intégrale|dérivée|probabilité|statistique|'
    r'formule|résoudre|démontrer|\d+[\+\-\*\/]\d+)\b', re.I
)
CODE_PATTERNS = re.compile(
    r'\b(code|python|programme|algorithme|implémenter|script|fonction|class|boucle)\b', re.I
)
SEARCH_PATTERNS = re.compile(
    r'\b(qu\'est-ce que|définition|expliquer|histoire|origine|qui est|quand|'
    r'wikipedia|recherche|trouver|synthèse|résumé)\b', re.I
)
FACT_PATTERNS = re.compile(
    r'\b(vrai|faux|vérifier|confirmer|source|référence|citation|preuve)\b', re.I
)


def compute_context_load(state: AcademicState) -> float:
    """Estimate context saturation [0, 1]."""
    msg_count = len(state.get("messages", []))
    query_len = len(state.get("user_query", ""))
    return min(1.0, (msg_count * 0.05) + (query_len / 2000))


def select_agents(state: AcademicState) -> RouterDecision:
    """
    Dynamic routing logic. Returns agents to activate and reasoning.
    Only selects agents that are registered.
    """
    query = state.get("user_query", "")
    context_load = compute_context_load(state)
    available = registry.list_agents()

    # Always start with planning (if registered)
    selected: List[str] = []
    reasoning_parts: List[str] = []

    # Planning always runs first
    if "planning" in available:
        selected.append("planning")
        reasoning_parts.append("planning systématique")

    # RAG: for explanations, definitions, research
    if "rag" in available and SEARCH_PATTERNS.search(query):
        selected.append("rag")
        reasoning_parts.append("recherche documentaire détectée")

    # Tools: for math or code
    if "tools" in available and (MATH_PATTERNS.search(query) or CODE_PATTERNS.search(query)):
        selected.append("tools")
        reasoning_parts.append("calcul/code détecté")

    # RAG fallback: if no specific signal, try RAG anyway for context
    if "rag" in available and "rag" not in selected and context_load < 0.7:
        selected.append("rag")
        reasoning_parts.append("RAG en fallback (contexte léger)")

    # Verification: always if confidence could be low or multiple agents active
    if "verification" in available and len(selected) > 1:
        selected.append("verification")
        reasoning_parts.append("vérification multi-sources")

    # Synthesis always runs last (if registered)
    if "synthesis" in available:
        selected.append("synthesis")
        reasoning_parts.append("synthèse finale")

    # Complexity estimate
    agent_count = len(selected)
    complexity = "low" if agent_count <= 2 else "medium" if agent_count <= 4 else "high"

    decision: RouterDecision = {
        "selected_agents": selected,
        "reasoning": " | ".join(reasoning_parts) or "routage par défaut",
        "estimated_complexity": complexity,
        "context_load": round(context_load, 3),
    }

    logger.info(
        f"[Router] Agents sélectionnés: {selected} | "
        f"Complexité: {complexity} | Charge: {context_load:.2f}"
    )
    return decision
