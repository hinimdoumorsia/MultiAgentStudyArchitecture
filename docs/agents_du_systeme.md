# 4. Agents du système

## Introduction — Comment fonctionne un agent dans ce système ?

Avant de détailler chaque agent, il est important de comprendre le **mécanisme commun** qui les gouverne tous. Chaque agent est une classe Python qui hérite de `BaseAgent`. Cette classe abstraite joue trois rôles simultanément :

1. **Elle est un nœud LangGraph** : grâce à la méthode `__call__`, chaque instance d'agent peut être directement branchée dans un graphe LangGraph sans aucun adaptateur.
2. **Elle est un observateur de performance** : elle mesure automatiquement la latence et le nombre de tokens consommés à chaque appel, sans que chaque agent ait à le faire lui-même.
3. **Elle est un filet de sécurité** : en cas d'erreur dans `process()`, elle ne laisse jamais le pipeline s'arrêter — elle reconstruit un état complet et cohérent avec des valeurs de fallback.

### Cycle de vie d'un appel agent

```
LangGraph appelle agent(state)
        │
        ▼
BaseAgent.__call__(state)          # point d'entrée universel
    │
    ├── start = time.perf_counter()  # démarrage du chronomètre
    │
    ├── result_state = self.process(state)
    │       │
    │       └── [logique métier de l'agent : LLM, ChromaDB, outils...]
    │
    ├── latency_ms = (perf_counter() - start) * 1000
    ├── tokens = self._extract_tokens(result_state)
    │
    ├── Construit AgentResult {
    │       agent_name, output, confidence,
    │       latency_ms, success=True, metadata, tokens
    │   }
    │
    ├── result_state["agent_results"].append(agent_result)
    │       # historique cumulatif de tous les agents déjà passés
    │
    └── return result_state          # transmis au nœud suivant du graphe
```

En cas d'exception dans `process()` :

```
Exception levée dans process()
        │
        ▼
BaseAgent.__call__ catch(exc)
    │
    ├── Construit un état de secours COMPLET
    │   (toutes les clés de AcademicState présentes avec valeurs vides)
    │   → évite les KeyError dans les agents suivants
    │
    ├── success = False dans AgentResult
    └── return état_de_secours      # le pipeline continue malgré l'erreur
```

Ce design est **critique** : si un agent échoue (timeout LLM, coupure réseau), les agents suivants reçoivent quand même un état valide et peuvent continuer à travailler avec ce qu'ils ont.

```python
# backend/agents/base.py — Code complet annoté

from abc import ABC, abstractmethod
from typing import Any, Dict
import time
import logging
from backend.state import AcademicState, AgentResult

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Classe de base abstraite pour tous les agents du système.

    HÉRITAGE : chaque agent spécialisé hérite de cette classe et
    n'a qu'une seule obligation : implémenter process().
    Tout le reste (timing, tokens, error handling, LangGraph compat)
    est géré ici une seule fois pour tous.

    ATTRIBUTS DE CLASSE (pas d'instance) :
    - name : identifiant court utilisé par le routeur et le registre
    - description : texte affiché dans l'UI et le endpoint /api/agents
    Ces deux attributs sont redéfinis dans chaque sous-classe.
    """
    name: str = "base_agent"
    description: str = "Abstract base agent"

    def __call__(self, state: AcademicState) -> Dict[str, Any]:
        """
        Méthode appelée automatiquement par LangGraph à chaque nœud.

        LangGraph s'attend à une signature : fn(state) -> state
        BaseAgent.__call__ respecte exactement ce contrat.

        Le wrapping ici fait trois choses invisibles pour les sous-classes :
        1. Chronomètre l'exécution avec perf_counter (précision µs)
        2. Standardise la structure de sortie (AgentResult)
        3. Intercepte toutes les exceptions sans stopper le pipeline
        """
        start = time.perf_counter()
        try:
            logger.info(f"[{self.name}] Processing: {state['user_query'][:80]}…")

            # ─── Appel au cœur métier de l'agent ────────────────────────
            result_state = self.process(state)

            # process() peut théoriquement retourner None (agent mal écrit)
            # on le remplace par un dict vide pour éviter une AttributeError
            if result_state is None:
                result_state = {}

            # ─── Mesure de performance ───────────────────────────────────
            latency_ms = (time.perf_counter() - start) * 1000

            # ─── Extraction des tokens consommés ─────────────────────────
            # _extract_tokens() cherche la clé "tokens" dans result_state
            # Si l'agent n'a pas renseigné cette clé, retourne des zéros
            tokens = self._extract_tokens(result_state)

            # ─── Construction du rapport d'exécution ─────────────────────
            # AgentResult est un TypedDict défini dans state.py
            # Il sert de "fiche de rapport" pour chaque agent
            agent_result: AgentResult = {
                "agent_name": self.name,
                # _extract_output() cherche dans l'ordre :
                # final_answer → plan → retrieved_docs → tool_results
                "output": self._extract_output(result_state),
                # _extract_confidence() lit verification_report si présent
                # sinon retourne 0.85 par défaut
                "confidence": self._extract_confidence(result_state),
                "latency_ms": round(latency_ms, 2),
                "success": True,
                # get_metadata() peut être surchargée par chaque agent
                # pour exposer des données spécifiques (ex: nb docs indexés)
                "metadata": self.get_metadata(state),
                "tokens": tokens,
            }

            # agent_results est une liste qui s'accumule au fil du pipeline
            # après PlanningAgent → [plan_result]
            # après RAGAgent      → [plan_result, rag_result]
            # après SynthesisAgent→ [plan_result, rag_result, ..., synth_result]
            if "agent_results" not in result_state:
                result_state["agent_results"] = []
            result_state["agent_results"].append(agent_result)

            logger.info(f"[{self.name}] Terminé en {latency_ms:.0f}ms | tokens: {tokens.get('total_tokens', 0)}")
            return result_state

        except Exception as exc:
            # ─── Gestion d'erreur : pipeline-safe ────────────────────────
            # L'état retourné ici est COMPLET :
            # toutes les clés que les agents suivants pourraient lire
            # sont présentes, avec les valeurs de l'état d'entrée.
            # → les agents suivants ne reçoivent jamais un état partiel
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[{self.name}] ERREUR: {exc}", exc_info=True)
            return {
                "errors": [f"{self.name}: {str(exc)}"],
                "agent_results": [{
                    "agent_name": self.name,
                    "output": "",
                    "confidence": 0.0,
                    "latency_ms": round(latency_ms, 2),
                    "success": False,
                    "metadata": {},
                    "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }],
                # Toutes les clés de AcademicState préservées depuis l'état entrant
                "plan":                state.get("plan", ""),
                "retrieved_docs":      state.get("retrieved_docs", ""),
                "tool_results":        state.get("tool_results", ""),
                "verification_report": state.get("verification_report", {}),
                "final_answer":        state.get("final_answer", ""),
                "user_query":          state.get("user_query", ""),
                "messages":            state.get("messages", []),
                "session_id":          state.get("session_id", ""),
                "router_decision":     state.get("router_decision", {}),
                "total_latency_ms":    state.get("total_latency_ms", 0.0),
                "run_id":              state.get("run_id", ""),
                "retry_count":         state.get("retry_count", 0),
            }

    @abstractmethod
    def process(self, state: AcademicState) -> Dict[str, Any]:
        """
        SEULE méthode obligatoire pour chaque agent.

        Convention de retour :
        - Retourner UNIQUEMENT les clés que cet agent modifie.
        - Ne pas retourner toutes les clés de AcademicState.
        - LangGraph fait une fusion (merge) de l'état existant avec le retour.

        Exemple : PlanningAgent retourne {"plan": "...", "tokens": {...}}
        Il ne retourne PAS retrieved_docs, tool_results, etc.
        """
        ...

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        """
        Peut être surchargée pour exposer des métriques agent-spécifiques.
        Ces données apparaissent dans AgentResult.metadata et sont visibles
        dans la réponse API sous agent_results[i].metadata.

        Exemple RAGAgent : {"indexed_documents": 42}
        Exemple ToolsAgent : {"available_tools": ["calculator", "wikipedia"]}
        """
        return {}

    # ── Helpers internes ─────────────────────────────────────────────────

    def _extract_output(self, result_state: Dict) -> str:
        """
        Cherche la sortie principale de l'agent dans result_state.
        Parcourt les clés dans l'ordre de priorité sémantique :
        final_answer > plan > retrieved_docs > tool_results
        Retourne "" si aucune clé n'est trouvée.
        """
        if result_state is None:
            return ""
        for key in ("final_answer", "plan", "retrieved_docs", "tool_results"):
            if result_state.get(key):
                v = result_state[key]
                return v if isinstance(v, str) else str(v)
        return ""

    def _extract_confidence(self, result_state: Dict) -> float:
        """
        Lit le score de confiance depuis verification_report si présent.
        Retourne 0.85 par défaut (confiance nominale avant vérification).
        VerificationAgent est le seul à renseigner cette clé.
        """
        if result_state is None:
            return 0.85
        report = result_state.get("verification_report", {})
        if report:
            return report.get("confidence_score", 0.8)
        return 0.85

    def _extract_tokens(self, result_state: Dict) -> Dict[str, int]:
        """
        Lit les tokens depuis result_state["tokens"].
        Chaque agent qui appelle un LLM doit mettre cette clé dans son retour.
        Si la clé est absente (agent sans LLM), retourne des zéros.

        Structure attendue :
        {
            "prompt_tokens":     int,  # tokens envoyés au LLM
            "completion_tokens": int,  # tokens générés par le LLM
            "total_tokens":      int   # somme des deux
        }
        """
        tokens = result_state.get("tokens", {})
        return {
            "prompt_tokens":     tokens.get("prompt_tokens", 0),
            "completion_tokens": tokens.get("completion_tokens", 0),
            "total_tokens":      tokens.get("total_tokens", 0)
        }
```

---

## 4.1 Agent de planification (`PlanningAgent`)

### Rôle et position dans le pipeline

`PlanningAgent` est le **premier agent exécuté** dans le pipeline, qu'il soit hiérarchique ou distribué. Son rôle est de transformer une question brute en un **plan d'action structuré** que les agents suivants vont utiliser comme feuille de route.

Concrètement, il répond à ces questions :
- De combien d'étapes a-t-on besoin pour répondre ?
- Quelle est la complexité de la question (low / medium / high) ?
- Faut-il chercher dans des documents (RAG) ?
- Faut-il exécuter des outils (calcul, code, Wikipedia) ?

### Pourquoi NVIDIA LLaMA 3.1-8b-instruct ?

La planification ne nécessite pas un modèle puissant : elle produit une sortie JSON courte (~200 tokens). Un modèle léger comme `meta/llama-3.1-8b-instruct` via l'API NVIDIA est suffisant et rapide. Le paramètre `temperature=0.1` est volontairement très bas pour forcer une sortie JSON **déterministe** : on ne veut pas de créativité ici, on veut toujours le même format.

### Mécanisme de contexte de session (A2A mémoire)

Avant même d'appeler le LLM, `PlanningAgent` consulte `memory_manager` pour récupérer l'historique de la session en cours. Si l'utilisateur a déjà posé des questions dans cette session, le plan tiendra compte du fil de la conversation. C'est une forme de **communication agent-à-agent via la mémoire partagée** plutôt que via l'état immédiat.

```
Session "session-001" : 3 échanges précédents
        │
        ▼
memory_manager.get_session_context("session-001")
        │
        ▼
Retourne : "Q: Qu'est-ce que Bayes ? R: Le théorème de Bayes..."
        │
        ▼
Injecté dans le prompt sous [HISTORIQUE RÉCENT]
→ Le LLM voit le contexte AVANT la question actuelle
→ Le plan résultant est cohérent avec la conversation
```

### Extraction JSON robuste

Le LLM retourne parfois du texte avant ou après le JSON (ex : "Voici le plan : {...} Voilà."). L'extraction par `re.search(r"\{.*\}", raw, re.DOTALL)` capture uniquement le bloc JSON même s'il est entouré de texte parasite. `re.DOTALL` permet au `.` de matcher les sauts de ligne, essentiel pour un JSON multi-lignes.

### Mécanisme de fallback par règles

Si le LLM est inaccessible (timeout, quota dépassé), `_fallback_plan()` prend le relais avec une détection par mots-clés. Ce n'est pas intelligent, mais c'est **suffisant pour ne pas bloquer le pipeline** : les agents suivants reçoivent quand même un plan minimal cohérent.

```python
# backend/agents/planning_agent.py — Code complet annoté

import os, json, re
from typing import Dict, Any
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from backend.agents.base import BaseAgent
from backend.state import AcademicState
from backend.memory.memory_manager import memory_manager

# ── Prompt système ────────────────────────────────────────────────────────
# La contrainte JSON est EXPLICITE dans le prompt.
# "Réponds TOUJOURS en JSON" → réduit les sorties libres du LLM.
# Le format exact est montré comme exemple → le LLM peut le copier/adapter.
# Pour les questions de code, un exemple spécifique est donné
# → le LLM génère un plan plus court et plus adapté.
SYSTEM_PROMPT = """Tu es un agent de planification pour un assistant académique.
Ta mission : analyser la question et produire un plan d'action structuré.

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

Pour les questions de CODE, sois plus concis :
{
  "plan_steps": ["Analyser la demande", "Générer le code", "Expliquer la solution"],
  "complexity": "medium",
  "requires_retrieval": false,
  "requires_tools": true,
  "estimated_tokens": 300,
  "strategy": "Génération de code Python avec explication"
}"""


class PlanningAgent(BaseAgent):
    name = "planning"
    description = "Décompose la requête en sous-tâches ordonnées et estime la complexité"

    def __init__(self, model: str = "meta/llama-3.1-8b-instruct"):
        """
        Initialisation du LLM de planification.

        Choix du modèle : meta/llama-3.1-8b-instruct via NVIDIA NIM
        → Modèle léger, rapide, suffisant pour générer un JSON court.
        → Plus économique en tokens que les grands modèles.

        max_tokens=512 : un plan JSON ne dépasse jamais 512 tokens.
        temperature=0.1 : quasi-déterministe → toujours le même format JSON.

        NOTE : le paramètre 'timeout' a été supprimé car l'API NVIDIA
        le rejette avec "Unsupported parameter(s): timeout".
        Le timeout est géré au niveau du décorateur @timeout dans ToolsAgent.
        """
        self.llm = ChatNVIDIA(
            model=model,
            api_key=os.getenv("NVIDIA_API_KEY"),
            max_tokens=512,
            temperature=0.1
        )

    def process(self, state: AcademicState) -> Dict[str, Any]:
        """
        Flux d'exécution complet du PlanningAgent :

        1. Lire l'historique de session depuis memory_manager
        2. Construire le prompt avec contexte optionnel
        3. Appeler le LLM NVIDIA
        4. Parser le JSON retourné (avec regex pour être robuste)
        5. Formater le plan en texte Markdown lisible
        6. Extraire les tokens consommés
        7. Retourner le plan + la complexité estimée
        """

        # ── Étape 1 : Contexte de session ────────────────────────────────
        # get_session_context() interroge SQLite pour les N derniers échanges
        # Si la session n'existe pas encore → retourne None ou ""
        contexte = memory_manager.get_session_context(state["session_id"])

        # ── Étape 2 : Construction du prompt avec contexte conditionnel ───
        # Le contexte est injecté AVANT la question pour que le LLM
        # comprenne la continuité de la conversation.
        if contexte:
            question_avec_contexte = (
                f"[HISTORIQUE RÉCENT]\n{contexte}\n\n"
                f"[QUESTION]\n{state['user_query']}"
            )
            print(f"[PlanningAgent] ✅ Contexte chargé ({len(contexte)} chars)")
        else:
            question_avec_contexte = state['user_query']
            print(f"[PlanningAgent] ❌ Aucun contexte — première question de la session")

        # Messages au format LangChain : SystemMessage + HumanMessage
        # SystemMessage = instructions permanentes du rôle de l'agent
        # HumanMessage  = la question spécifique à traiter
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question : {question_avec_contexte}"),
        ]

        # ── Étape 3 : Appel LLM avec fallback intégré ────────────────────
        try:
            response = self.llm.invoke(messages)
            raw = response.content          # texte brut retourné par le LLM
        except Exception as e:
            print(f"[PlanningAgent] ⚠️ LLM inaccessible: {e} → fallback par règles")
            # Si le LLM échoue, on utilise la détection par mots-clés
            # pour générer un plan minimal qui débloque le pipeline
            return self._fallback_plan(state)

        # ── Étape 4 : Extraction JSON robuste ────────────────────────────
        # re.DOTALL permet à "." de matcher les sauts de ligne
        # → capture le JSON même s'il est sur plusieurs lignes
        # Exemple de cas traité :
        #   raw = "Voici le plan :\n{\n  \"plan_steps\": [...]\n}\nJ'espère..."
        #   match.group() = "{\n  \"plan_steps\": [...]\n}"
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                plan_data = json.loads(match.group())
            except json.JSONDecodeError:
                # JSON malformé malgré le regex → fallback
                plan_data = {}
        else:
            plan_data = {}

        # Si le LLM n'a pas fourni d'étapes, on passe au fallback
        if not plan_data.get("plan_steps"):
            return self._fallback_plan(state)

        # ── Étape 5 : Formatage Markdown du plan ─────────────────────────
        # Le plan_text est stocké dans AcademicState["plan"]
        # et sera lu par RAGAgent, ToolsAgent et VerificationAgent
        # via l'état partagé (communication A2A implicite).
        plan_text = (
            f"**Stratégie :** {plan_data.get('strategy', 'Analyse directe')}\n"
            f"**Complexité :** {plan_data.get('complexity', 'medium')}\n"
            f"**Étapes :**\n"
            + "\n".join(
                f"  {i+1}. {step}"
                for i, step in enumerate(plan_data.get("plan_steps", []))
            )
        )

        # ── Étape 6 : Extraction des tokens consommés ────────────────────
        # L'API NVIDIA retourne les tokens dans response.response_metadata
        # sous la clé 'token_usage' si la facturation est activée.
        # Sinon on estime : ~4 caractères ≈ 1 token (règle empirique GPT)
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            tu = response.response_metadata['token_usage']
            tokens_data = {
                "prompt_tokens":     tu.get('prompt_tokens', 0),
                "completion_tokens": tu.get('completion_tokens', 0),
                "total_tokens":      tu.get('total_tokens', 0)
            }
        else:
            tokens_data = {
                "prompt_tokens":     len(str(messages)) // 4,
                "completion_tokens": len(raw) // 4,
                "total_tokens":      (len(str(messages)) + len(raw)) // 4
            }

        # ── Étape 7 : Retour partiel de l'état ───────────────────────────
        # On retourne UNIQUEMENT les clés que cet agent a modifiées.
        # LangGraph merge ce dict avec l'état existant.
        # router_decision est enrichi avec la complexité estimée
        # → le routeur peut l'utiliser pour des décisions futures.
        return {
            "plan": plan_text,
            "router_decision": {
                **state.get("router_decision", {}),  # préserver les clés existantes
                "estimated_complexity": plan_data.get("complexity", "medium"),
            },
            "tokens": tokens_data
        }

    def _fallback_plan(self, state: AcademicState) -> Dict[str, Any]:
        """
        Plan de secours basé sur la détection de mots-clés.

        Déclenché dans deux cas :
        1. Le LLM NVIDIA est inaccessible (timeout, erreur réseau, quota)
        2. La réponse du LLM ne contient pas de plan_steps valides

        La détection est volontairement simple : any() + liste de mots-clés.
        L'objectif n'est pas d'être précis, mais de fournir UN plan
        qui permette aux agents suivants de continuer leur travail.

        Complexités retournées :
        - "low"    : questions de calcul simple → ToolsAgent sera prioritaire
        - "medium" : questions de code ou générales → pipeline standard
        """
        query = state["user_query"]

        if any(w in query.lower() for w in ["code", "python", "implémenter", "programme", "fonction"]):
            plan_text = (
                "**Stratégie :** Génération de code\n"
                "**Complexité :** medium\n"
                "**Étapes :**\n"
                "  1. Analyser la demande de code\n"
                "  2. Générer une solution\n"
                "  3. Expliquer le code"
            )
            complexity = "medium"

        elif any(w in query.lower() for w in ["somme", "calcul", "math", "équation", "résoudre"]):
            plan_text = (
                "**Stratégie :** Calcul direct\n"
                "**Complexité :** low\n"
                "**Étapes :**\n"
                "  1. Effectuer le calcul\n"
                "  2. Fournir le résultat"
            )
            complexity = "low"

        else:
            plan_text = (
                "**Stratégie :** Analyse et réponse\n"
                "**Complexité :** medium\n"
                "**Étapes :**\n"
                "  1. Comprendre la question\n"
                "  2. Rechercher l'information\n"
                "  3. Synthétiser la réponse"
            )
            complexity = "medium"

        # tokens à 0 car aucun LLM n'a été appelé
        return {
            "plan": plan_text,
            "router_decision": {
                **state.get("router_decision", {}),
                "estimated_complexity": complexity,
            },
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        """Expose la longueur de la requête dans les métriques."""
        return {"query_length": len(state["user_query"])}
```

---

## 4.2 Agent RAG — Retrieval-Augmented Generation (`RAGAgent`)

### Rôle et position dans le pipeline

`RAGAgent` est le **deuxième agent** du pipeline. Son rôle est d'enrichir la réponse avec des **documents académiques indexés** dans ChromaDB. Il reçoit la question originale ET le plan produit par `PlanningAgent` (via l'état partagé), les combine pour former une requête de recherche enrichie, puis synthétise les passages les plus pertinents avec Groq.

Sans documents indexés, il bascule sur les connaissances générales du LLM — le pipeline ne s'arrête jamais.

### Pourquoi ChromaDB + Groq ?

ChromaDB stocke les documents sous forme de **vecteurs d'embeddings** (représentation numérique du sens du texte). Quand une question arrive, ChromaDB convertit aussi la question en vecteur et cherche les documents dont le vecteur est le plus proche (similarité cosinus). Cela permet de trouver des documents pertinents même si les mots exacts ne correspondent pas.

Groq est choisi pour la synthèse car il est **très rapide** (~500 tokens/seconde), adapté à la génération de réponses documentaires de longueur moyenne.

### Communication A2A via l'état partagé

```
PlanningAgent retourne : {"plan": "**Stratégie :** Recherche documentaire\n**Étapes :**..."}
                                          │
                                          ▼ (stocké dans AcademicState["plan"])
RAGAgent.process(state) lit : plan = state.get("plan", "")
                                          │
                                          ▼
enriched_query = question + plan[:300]
                                          │
                          → Requête de recherche plus précise dans ChromaDB
```

En ajoutant le contexte du plan à la requête de recherche, RAGAgent cherche des documents qui correspondent non seulement à la question, mais aussi à la **stratégie de résolution** choisie par PlanningAgent.

### Gestion des rate limits Groq (3 tentatives)

L'API Groq a des limites de requêtes par minute (RPM). En cas de dépassement, elle retourne une erreur 429. RAGAgent implémente une **boucle de retry avec attente** (3 tentatives, 5 secondes entre chaque). Si les 3 tentatives échouent, il retourne un message d'erreur explicite sans bloquer le pipeline.

```python
# backend/agents/rag_agent.py — Code complet annoté

from typing import Dict, Any, List
import os, logging, time, uuid

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from backend.agents.base import BaseAgent
from backend.state import AcademicState

logger = logging.getLogger(__name__)

# Le prompt RAG indique explicitement au LLM comment traiter l'absence de documents.
# "reste factuel" → réduit les hallucinations dans les réponses documentaires.
SYSTEM_PROMPT = """Tu es un agent de recherche documentaire académique (RAG).
Tu reçois une question et un contexte optionnel de documents.
Synthétise les informations pertinentes trouvées pour répondre à la question.
Si aucun document n'est disponible, indique-le clairement et propose
une réponse basée sur tes connaissances. Sois précis, cite tes sources
quand disponibles, et reste factuel."""


class RAGAgent(BaseAgent):
    name = "rag"
    description = (
        "Recherche vectorielle dans ChromaDB + synthèse Groq. "
        "Retourne les passages les plus pertinents avec leur score de similarité."
    )

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """
        Initialise Groq LLM + ChromaDB.

        llama-3.1-8b-instant : modèle rapide de Groq, optimisé pour
        les réponses courtes-moyennes (~512 tokens). Parfait pour
        synthétiser des passages documentaires.

        temperature=0.1 : réponses factuelles, peu de variation
        max_tokens=512  : suffisant pour une synthèse documentaire
        """
        self.llm = ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=512
        )
        self.model = model
        self._chroma = None
        self._collection = None
        self._init_vector_store()

    def _init_vector_store(self):
        """
        Initialise la connexion à ChromaDB en mode persistant.

        PersistentClient : les données survivent aux redémarrages du serveur.
        Le dossier est créé automatiquement s'il n'existe pas.

        hnsw:space = "cosine" : algorithme de recherche approximative (HNSW)
        avec métrique cosinus. La similarité cosinus mesure l'angle entre
        deux vecteurs (0 = opposés, 1 = identiques) indépendamment
        de leur magnitude → idéal pour comparer des textes de longueurs variées.

        get_or_create_collection : idempotent → sûr de l'appeler plusieurs fois.

        Fallback gracieux : si ChromaDB est absent (première installation,
        dépendance manquante), self._collection reste None et _search_documents
        retourne [] sans erreur.
        """
        try:
            import chromadb
            db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
            os.makedirs(db_path, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=db_path)
            self._collection = self._chroma.get_or_create_collection(
                name="academic_docs",
                metadata={"hnsw:space": "cosine"},  # métrique de similarité
            )
            logger.info(f"[RAGAgent] ChromaDB initialisé à {db_path}")
        except Exception as e:
            logger.warning(f"[RAGAgent] ChromaDB indisponible ({e}) → mode fallback")

    def _search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Effectue une recherche vectorielle dans ChromaDB.

        Fonctionnement interne de ChromaDB.query() :
        1. ChromaDB convertit 'query' en vecteur via son modèle d'embedding interne
        2. Il compare ce vecteur avec tous les vecteurs stockés (HNSW index)
        3. Retourne les n_results plus proches avec leurs distances cosinus

        min(n_results, collection.count()) : évite l'erreur si la collection
        contient moins de n_results documents.

        La distance cosinus [0,1] est convertie en score de similarité [0,1] :
        similarity = 1 - distance
        → distance 0.0 = documents identiques → similarity 1.0
        → distance 0.5 = documents moyennement liés → similarity 0.5

        Retourne [] si ChromaDB est absent ou si une erreur survient.
        """
        if self._collection is None:
            return []
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self._collection.count() or 1),
            )
            docs = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                dist = results.get("distances", [[]])[0][i] if results.get("distances") else 0
                docs.append({
                    "content":    doc,
                    "source":     meta.get("source", "Document inconnu"),
                    "similarity": round(1 - dist, 3),  # distance → similarité
                })
            return docs
        except Exception as e:
            logger.warning(f"[RAGAgent] Erreur de recherche: {e}")
            return []

    def add_document(self, content: str, source: str, doc_id: str = None):
        """
        Indexe un nouveau document dans ChromaDB.

        Appelé depuis l'endpoint POST /api/documents de l'API FastAPI.
        ChromaDB génère automatiquement l'embedding du contenu.
        doc_id = UUID aléatoire si non fourni → garantit l'unicité.

        Après ajout, le document est immédiatement disponible pour la recherche
        (pas de reindexation nécessaire, ChromaDB maintient l'index HNSW
        de façon incrémentale).
        """
        if self._collection is None:
            raise RuntimeError("ChromaDB non disponible")
        self._collection.add(
            documents=[content],
            metadatas=[{"source": source}],
            ids=[doc_id or str(uuid.uuid4())],
        )

    def process(self, state: AcademicState) -> Dict[str, Any]:
        """
        Flux d'exécution de RAGAgent :

        1. Lire la question + le plan depuis l'état partagé
        2. Enrichir la requête de recherche avec le contexte du plan (A2A)
        3. Rechercher les documents pertinents dans ChromaDB
        4. Construire le prompt selon la disponibilité des documents
        5. Appeler Groq avec retry (3 tentatives, 5s entre chaque)
        6. Retourner la synthèse + métadonnées de sources
        """

        query = state["user_query"]
        # Lecture du plan produit par PlanningAgent via l'état partagé
        # C'est ici que se manifeste la communication A2A (Agent-to-Agent)
        plan = state.get("plan", "")
        plan_text = plan if plan is not None else ""

        # ── Enrichissement de la requête (A2A via état partagé) ──────────
        # Les 300 premiers caractères du plan suffisent pour orienter
        # la recherche sans alourdir le vecteur de requête.
        enriched_query = (
            f"{query}\nContexte du plan : {plan_text[:300]}"
            if plan_text else query
        )

        # ── Recherche vectorielle dans ChromaDB ──────────────────────────
        docs = self._search_documents(enriched_query)

        # ── Construction du prompt selon disponibilité des documents ──────
        if docs:
            # Des documents ont été trouvés → on les inclut comme contexte
            # Format : [Source: X | Similarité: Y]\nContenu du document
            # Le score de similarité aide le LLM à pondérer les sources
            context = "\n\n".join(
                f"[Source: {d['source']} | Similarité: {d['similarity']}]\n{d['content']}"
                for d in docs
            )
            prompt = f"Question : {query}\n\nDocuments trouvés :\n{context}"
        else:
            # Aucun document → on indique clairement au LLM de répondre
            # depuis ses connaissances générales (plutôt que d'inventer des sources)
            prompt = (
                f"Question : {query}\n\nAucun document spécifique trouvé "
                "dans la base. Réponds en utilisant tes connaissances académiques générales."
            )

        # ── Appel Groq avec retry (gestion rate limits 429) ─────────────
        # Groq limite à ~30 RPM sur le plan gratuit.
        # 3 tentatives avec 5s d'attente couvrent la plupart des cas de throttling.
        response = None
        for attempt in range(3):
            try:
                response = self.llm.invoke(prompt)
                break  # succès → on sort de la boucle
            except Exception as e:
                logger.warning(f"[RAGAgent] Tentative {attempt+1}/3 échouée: {e}")
                if attempt < 2:
                    time.sleep(5)   # pause avant le prochain essai
                else:
                    # 3 échecs consécutifs → on abandonne proprement
                    return {
                        "retrieved_docs": (
                            "Service RAG temporairement indisponible. "
                            "Veuillez réessayer dans quelques instants."
                        ),
                        "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }

        # ── Métadonnées de sources affichées dans la réponse ─────────────
        sources_info = (
            f"\n\n📚 **Sources consultées :** {len(docs)} document(s)"
            if docs
            else "\n\n📚 **Sources :** Connaissances générales (aucun document indexé)"
        )

        # ── Estimation des tokens ─────────────────────────────────────────
        # Groq ne retourne pas systématiquement l'usage dans response_metadata.
        # L'estimation par longueur de texte est fiable à ±20%.
        response_content = response.content
        tokens_data = {
            "prompt_tokens":     len(prompt) // 4,
            "completion_tokens": len(response_content) // 4,
            "total_tokens":      (len(prompt) + len(response_content)) // 4
        }

        return {
            "retrieved_docs": response_content + sources_info,
            "tokens": tokens_data
        }

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        """Expose le nombre de documents indexés dans ChromaDB."""
        count = 0
        if self._collection:
            try:
                count = self._collection.count()
            except Exception:
                pass
        return {"indexed_documents": count}
```

---

## 4.3 Agent d'exécution d'outils (`ToolsAgent`)

### Rôle et position dans le pipeline

`ToolsAgent` est le **troisième agent** du pipeline. Il sélectionne et exécute des outils externes adaptés à la question. Sa particularité : il n'appelle PAS directement un outil — il demande d'abord au LLM de **choisir quel outil utiliser et avec quel paramètre**, puis exécute ce choix.

Ce double niveau (LLM → décision → exécution outil) correspond au pattern **ReAct** (Reasoning + Acting) : l'agent raisonne sur quoi faire avant d'agir.

### Le décorateur `@timeout` — Comment ça marche

Chaque outil est enveloppé dans un décorateur `@timeout` qui l'exécute dans un **thread séparé via `ThreadPoolExecutor`**. Si l'outil dépasse son délai, `future.result(timeout=N)` lève `FuturesTimeoutError`, le thread est abandonné, et le décorateur retourne la valeur `default_return` sans lever d'exception.

```
ToolsAgent._decide_tools() → [{"tool": "calculator", "input": "2**32"}]
        │
        ▼
calculator("2**32")             ← appel de l'outil
        │
        ▼ (via @timeout(seconds=10))
ThreadPoolExecutor.submit(calculator, "2**32")
        │
        ├── Si résultat en < 10s → retourne "Résultat : 4294967296"
        └── Si dépasse 10s      → retourne "Erreur: calcul trop long (>10s)"
```

### Sandbox de sécurité pour `calculator`

`eval()` est utilisé mais avec un **sandbox strict** : `{"__builtins__": {}}` bloque tous les builtins Python (import, open, exec...). Seules les fonctions du module `math` et quelques builtins inoffensifs (abs, round, sum, min, max) sont autorisés.

### Isolation par subprocess pour `python_executor`

L'exécution de code Python arbitraire est **dangereuse par nature**. `python_executor` utilise `subprocess.run()` pour exécuter le code dans un **processus complètement séparé** du serveur FastAPI. Si le code plante, le processus enfant meurt mais le serveur continue. Trois couches de protection :
1. Validation syntaxique via `ast.parse()` avant exécution
2. Blocage de mots-clés dangereux (os, sys, subprocess, open...)
3. Limite de taille (100 lignes, 5000 caractères)

### Intégration MCP (Model Context Protocol)

En plus des 5 outils locaux, `ToolsAgent` charge dynamiquement les outils du serveur MCP (`mcp_server.list_tools()`). MCP est un protocole standard pour exposer des outils externes à des LLM agents. Si un outil MCP est disponible (ex: outil de recherche institutionnelle), il est automatiquement fusionné dans `ALL_TOOLS` et proposé au LLM dans son prompt.

```python
# backend/agents/tools_agent.py — Code complet annoté

import math, logging, ast, re, json, os, subprocess, tempfile
from typing import Dict, Any, List
from functools import wraps
from concurrent.futures import TimeoutError as FuturesTimeoutError, ThreadPoolExecutor

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from backend.agents.base import BaseAgent
from backend.state import AcademicState
from backend.mcp import mcp_server

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# DÉCORATEUR @timeout — Protection contre les outils bloquants
# ════════════════════════════════════════════════════════════════════════

def timeout(seconds: int, default_return: str = "Timeout: opération trop longue"):
    """
    Décorateur qui limite le temps d'exécution d'une fonction.

    Mécanisme :
    - La fonction est soumise à un ThreadPoolExecutor (1 thread)
    - future.result(timeout=seconds) attend N secondes maximum
    - Si le délai est dépassé : FuturesTimeoutError est capturée
      et default_return est retourné sans propager l'exception

    Avantage vs signal.alarm() : fonctionne sur Windows ET Linux.
    signal.alarm() est UNIX-only → incompatible avec certains déploiements.

    Le thread continue en arrière-plan après timeout (Python ne peut pas
    tuer un thread de force), mais le pipeline n'est pas bloqué.
    """
    def decorator(func):
        @wraps(func)                # préserve __name__, __doc__, etc.
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except FuturesTimeoutError:
                    logger.warning(f"[ToolsAgent] Timeout {seconds}s dépassé: {func.__name__}")
                    return default_return
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════
# OUTIL 1 : Calculatrice sécurisée
# ════════════════════════════════════════════════════════════════════════

@tool
@timeout(seconds=10, default_return="Erreur: calcul trop long (>10s)")
def calculator(expression: str) -> str:
    """
    Évalue une expression mathématique dans un sandbox sécurisé.

    Sandbox : eval() avec __builtins__={} bloque TOUS les builtins Python.
    Seules les fonctions math.* et {abs, round, sum, min, max} sont autorisées.

    Exemples valides : '2**10', 'math.sqrt(144)', 'sum([1,2,3])', 'abs(-5)'
    Exemples rejetés : 'import os', '__import__("os")', 'open("file")'

    La limite de 500 caractères prévient les expressions pathologiques
    conçues pour saturer la mémoire (ex: '9'*500000).
    """
    try:
        if len(expression) > 500:
            return "Erreur: expression trop longue (>500 caractères)"

        # Construction du namespace sécurisé
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "sum": sum, "min": min, "max": max})

        # eval avec namespace restreint — __builtins__={} = pas d'imports possible
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Résultat : {result}"
    except Exception as e:
        return f"Erreur de calcul : {e}"


# ════════════════════════════════════════════════════════════════════════
# OUTIL 2 : Exécuteur Python sécurisé
# ════════════════════════════════════════════════════════════════════════

@tool
@timeout(seconds=20, default_return="Erreur: exécution Python trop longue (>20s)")
def python_executor(code: str) -> str:
    """
    Exécute du code Python dans un sous-processus isolé.

    Isolation par subprocess : le code s'exécute dans un NOUVEAU processus
    Python, complètement séparé du processus FastAPI. Si le code plante
    (segfault, MemoryError, sys.exit()), le serveur principal n'est pas affecté.

    Flux d'exécution :
    1. Vérification de taille (≤100 lignes, ≤5000 chars)
    2. Validation syntaxique via ast.parse() → erreur propre si invalide
    3. Blocage des imports dangereux (liste de mots-clés interdits)
    4. Écriture dans un fichier temporaire .py
    5. subprocess.run(['python', temp_file], timeout=20)
    6. Lecture de stdout/stderr
    7. Suppression du fichier temporaire

    Limites intentionnelles :
    - Pas d'accès au filesystem (import os bloqué)
    - Pas d'accès réseau (import requests bloqué implicitement par absence d'os)
    - Pas de sous-processus (subprocess bloqué)
    """
    # ── Validation de taille ──────────────────────────────────────────────
    if len(code) > 5000:
        return "Erreur: code trop long (>5000 caractères). Limitez votre code."
    if code.count('\n') > 100:
        return "Erreur: code trop long (>100 lignes). Simplifiez votre code."

    # ── Validation syntaxique ─────────────────────────────────────────────
    # ast.parse() lève SyntaxError si le code est syntaxiquement invalide
    # → meilleur message d'erreur que l'erreur subprocess brute
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"Erreur : Code Python invalide - {e}"

    # ── Blocage des imports dangereux ─────────────────────────────────────
    forbidden = [
        "import os", "import sys", "open(", "exec(", "eval(",
        "__import__", "subprocess", "shutil", "importlib"
    ]
    if any(f in code for f in forbidden):
        return "Erreur : code refusé pour des raisons de sécurité."

    # ── Exécution dans un fichier temporaire ─────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,    # capture stdout ET stderr
            text=True,              # retourne des str (pas bytes)
            timeout=20              # timeout subprocess natif (double sécurité)
        )

        os.unlink(temp_file)        # nettoyage systématique du fichier temp

        if result.stderr:
            return f"Erreur d'exécution : {result.stderr[:500]}"
        return result.stdout if result.stdout else "Exécution réussie (aucun output)"

    except subprocess.TimeoutExpired:
        return "Erreur : exécution Python trop longue (>20s). Simplifiez votre code."
    except Exception as e:
        return f"Erreur d'exécution : {e}"


# ════════════════════════════════════════════════════════════════════════
# OUTIL 3 : Recherche Wikipedia
# ════════════════════════════════════════════════════════════════════════

@tool
@timeout(seconds=8, default_return="Erreur: recherche Wikipedia trop longue (>8s)")
def wikipedia_search(query: str) -> str:
    """
    Interroge l'API REST Wikipedia sans dépendance externe (urllib only).

    Stratégie FR → EN : essaie d'abord Wikipedia français.
    Si la page n'existe pas ou n'a pas d'extrait, tente Wikipedia anglais.
    → Couvre la majorité des questions académiques.

    API utilisée : /api/rest_v1/page/summary/{titre}
    Elle retourne un JSON avec 'title', 'extract', 'thumbnail'...
    On prend uniquement 'extract' tronqué à 1000 chars.

    User-Agent personnalisé : bonne pratique pour les APIs Wikipedia
    (identification de l'application appelante).

    urllib.parse.quote() : encode les espaces et caractères spéciaux
    pour les inclure dans l'URL (ex: "théorème de Bayes" → "th%C3%A9or%C3%A8me%20de%20Bayes")
    """
    try:
        import urllib.request, urllib.parse

        if len(query) > 200:
            query = query[:200]     # limite la taille de l'URL

        encoded = urllib.parse.quote(query)
        for lang in ["fr", "en"]:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AcademicMAS/1.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data.get("extract"):
                        extract = data['extract'][:1000]
                        return f"[Wikipedia {lang.upper()}] {data['title']}:\n{extract}"
            except Exception:
                continue            # passe au prochain lang en cas d'erreur
        return "Aucun résultat Wikipedia trouvé."
    except Exception as e:
        return f"Erreur Wikipedia : {e}"


# ════════════════════════════════════════════════════════════════════════
# OUTILS 4 & 5 : Formateurs (LaTeX, Citations)
# ════════════════════════════════════════════════════════════════════════

@tool
@timeout(seconds=5, default_return="Erreur: formatage trop long")
def latex_formatter(expression: str) -> str:
    """Enveloppe l'expression dans des balises LaTeX display math ($$...$$)."""
    safe_expr = expression.replace('_', '\\_').replace('&', '\\&')
    return f"$${safe_expr}$$"


@tool
@timeout(seconds=5, default_return="Erreur: formatage citation trop long")
def citation_formatter(citation: str) -> str:
    """Formate une citation en style APA simple."""
    return f"📖 **Citation (APA)** : *{citation}*"


# ════════════════════════════════════════════════════════════════════════
# AGRÉGATION DES OUTILS (locaux + MCP)
# ════════════════════════════════════════════════════════════════════════

# Outils locaux disponibles par défaut
AVAILABLE_TOOLS = {
    "calculator":       calculator,
    "python_executor":  python_executor,
    "wikipedia_search": wikipedia_search,
    "latex_formatter":  latex_formatter,
    "citation_formatter": citation_formatter,
}

# Chargement dynamique des outils MCP
# mcp_server expose des outils tiers via le protocole MCP (Model Context Protocol)
# Si aucun outil MCP n'est disponible, MCP_TOOLS reste {}
MCP_TOOLS = {}
try:
    for tool_name in mcp_server.list_tools():
        MCP_TOOLS[tool_name] = lambda x, tn=tool_name: mcp_server.call_tool(tn, x)
    if MCP_TOOLS:
        logger.info(f"[ToolsAgent] Outils MCP chargés: {list(MCP_TOOLS.keys())}")
except Exception as e:
    logger.warning(f"[ToolsAgent] MCP indisponible: {e}")

# Fusion : les outils MCP enrichissent les outils locaux
ALL_TOOLS = {**AVAILABLE_TOOLS, **MCP_TOOLS}


class ToolsAgent(BaseAgent):
    name = "tools"
    description = (
        "Exécution d'outils : calculator(10s), python(20s), wikipedia(8s), "
        "latex, citations. + outils MCP dynamiques."
    )

    def __init__(self, model: str = "meta/llama-3.1-8b-instruct"):
        """
        Le LLM ici joue le rôle de "sélecteur d'outils" : il reçoit
        la question + le plan, et retourne un JSON indiquant quel outil
        appeler avec quel paramètre. C'est le pattern ReAct.

        NVIDIA LLaMA 3.1-8b : même modèle que PlanningAgent.
        max_tokens=512 : un JSON de sélection d'outils est court.
        """
        self.llm = ChatNVIDIA(
            model=model,
            api_key=os.getenv("NVIDIA_API_KEY"),
            max_tokens=512,
            temperature=0.1
        )
        self.tools = dict(ALL_TOOLS)

    def _decide_tools(self, query: str, plan: str) -> List[Dict]:
        """
        Demande au LLM de choisir les outils appropriés.

        Le LLM reçoit :
        - La description de chaque outil disponible (dans SYSTEM_PROMPT)
        - La question de l'utilisateur (tronquée à 500 chars)
        - Le plan de PlanningAgent (tronqué à 400 chars)

        Il doit retourner un JSON :
        {"tools_to_use": [{"tool": "calculator", "input": "2**32"}], "reasoning": "..."}

        Si le JSON est invalide ou si aucun outil n'est nécessaire :
        {"tools_to_use": [], "reasoning": "pas besoin d'outil"}
        → process() retournera "Aucun outil externe requis"
        """
        SYSTEM_PROMPT = f"""Tu es un agent d'exécution d'outils académiques.

RÈGLES :
- Pour python_executor, fournis du code Python VALIDE (pas une description)
- Ne dépasse pas 100 lignes de code
- Utilise calculator pour les maths simples, wikipedia pour les faits

Outils disponibles :
- calculator(expression) : math (max 500 chars)
- python_executor(code)  : Python (max 100 lignes)
- wikipedia_search(query): Wikipedia (max 200 chars)
- latex_formatter(expr)  : formate en LaTeX
- citation_formatter(c)  : formate une citation APA

Réponds UNIQUEMENT en JSON :
{{"tools_to_use": [{{"tool": "nom", "input": "paramètre"}}], "reasoning": "pourquoi"}}
Si aucun outil : {{"tools_to_use": [], "reasoning": "pas besoin"}}"""

        prompt = f"Question : {query[:500]}\nPlan : {plan[:400]}"
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        try:
            response = self.llm.invoke(messages)
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("tools_to_use", [])
        except Exception as e:
            logger.warning(f"[ToolsAgent] Erreur décision: {e}")
        return []

    def process(self, state: AcademicState) -> Dict[str, Any]:
        """
        Flux d'exécution de ToolsAgent :

        1. Lire question + plan depuis l'état partagé
        2. Demander au LLM de sélectionner les outils (ReAct)
        3. Valider chaque appel (sécurité supplémentaire pour python_executor)
        4. Exécuter les outils (avec timeouts intégrés)
        5. Tronquer les résultats à 2000 chars (évite de saturer l'état)
        6. Retourner la concaténation des résultats
        """
        query = state["user_query"]
        plan = state.get("plan", "")
        tool_calls = self._decide_tools(query, plan)

        if not tool_calls:
            return {"tool_results": "Aucun outil externe requis pour cette question."}

        results = []
        for call in tool_calls:
            tool_name  = call.get("tool", "")
            tool_input = call.get("input", "")

            # ── Validation supplémentaire pour python_executor ───────────
            # Le LLM peut parfois envoyer une description en langage naturel
            # au lieu de vrai code Python → on le rejette proprement
            if tool_name == "python_executor":
                code_indicators = ["print", "=", "+", "-", "*", "/", "for",
                                   "while", "if", "def", "class", "return"]
                if not any(ind in tool_input for ind in code_indicators):
                    results.append(f"⚠️ **{tool_name}** ignoré: ce n'est pas du code Python")
                    continue
                try:
                    ast.parse(tool_input)   # validation syntaxique ici aussi
                except SyntaxError as e:
                    results.append(f"⚠️ **{tool_name}** ignoré: erreur syntaxe - {e}")
                    continue

            # ── Exécution de l'outil ──────────────────────────────────────
            if tool_name in self.tools:
                logger.info(f"[ToolsAgent] Appel de {tool_name}()")
                try:
                    result = self.tools[tool_name](tool_input)
                    # Troncature à 2000 chars : évite de saturer AcademicState
                    # et de dépasser le contexte de SynthesisAgent
                    if len(result) > 2000:
                        result = result[:2000] + "... (résultat tronqué)"
                    results.append(f"🔧 **{tool_name}** → {result}")
                except Exception as e:
                    results.append(f"🔧 **{tool_name}** → Erreur: {str(e)[:200]}")
            else:
                results.append(f"⚠️ Outil inconnu : {tool_name}")

        return {"tool_results": "\n\n".join(results) if results else "Aucun outil exécuté"}

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        """Expose la liste des outils disponibles dans les métriques."""
        return {"available_tools": list(self.tools.keys())}
```

---

## 4.4 Agent de vérification (`VerificationAgent`)

### Rôle et position dans le pipeline

`VerificationAgent` est le **quatrième agent**. Il reçoit les sorties de TOUS les agents précédents (plan, documents RAG, résultats d'outils) et produit un **rapport de qualité** sous forme de JSON structuré. Ce rapport est ensuite lu par `SynthesisAgent` pour adapter sa réponse finale.

C'est l'agent "critique" du système : il ne génère pas de contenu, il **évalue** le contenu généré.

### Ce que le score de confiance signifie concrètement

| Score | Interprétation | Action de SynthesisAgent |
|-------|----------------|--------------------------|
| > 0.8 | Informations cohérentes, sources fiables | Synthèse normale |
| 0.6–0.8 | Quelques incertitudes | Synthèse avec nuances |
| < 0.6 | Incohérences détectées ou manque de sources | Avertissement ⚠️ affiché |
| Recommandation "FALLBACK" | Informations trop peu fiables | Réponse directe sans contexte |

### Gestion du rate limiting Groq

`VerificationAgent` introduit une pause de **3 secondes** avant son appel LLM. Cette pause est nécessaire car `RAGAgent` vient juste de faire un appel Groq. Sans cette pause, les deux appels consécutifs déclenchent régulièrement l'erreur 429 (trop de requêtes). C'est un compromis entre latence et fiabilité.

```python
# backend/agents/verification_agent.py — Code complet annoté

from typing import Dict, Any
import json, re, logging, time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from backend.agents.base import BaseAgent
from backend.state import AcademicState

logger = logging.getLogger(__name__)

# Le prompt force une sortie JSON stricte avec des champs bien définis.
# "UNIQUEMENT en JSON" : réduit le risque de texte parasite autour du JSON.
# Les champs couvrent 4 dimensions d'évaluation :
# - cohérence (consistency_check)
# - fiabilité (confidence_score, quality_score)
# - problèmes détectés (potential_hallucinations, missing_information)
# - recommandation actionnable (recommendation + notes)
SYSTEM_PROMPT = """Tu es un agent de vérification académique rigoureux.
Tu reçois : la question originale, le plan, les documents RAG, et les résultats d'outils.
Ton rôle : vérifier la cohérence, détecter les incohérences, produire un rapport.

Réponds UNIQUEMENT en JSON :
{
  "confidence_score": 0.0-1.0,
  "consistency_check": "Cohérent|Incohérent|Partiel",
  "potential_hallucinations": ["liste d'affirmations douteuses"],
  "missing_information": ["ce qui manque"],
  "quality_score": 0.0-1.0,
  "recommendation": "PROCEED|RETRY|FALLBACK",
  "verification_notes": "commentaire court"
}"""


class VerificationAgent(BaseAgent):
    name = "verification"
    description = (
        "Vérifie la cohérence des sorties, détecte les hallucinations "
        "potentielles, produit un score de confiance [0,1]."
    )

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.llm = ChatGroq(model=model, max_tokens=1024, temperature=0.1)

    def process(self, state: AcademicState) -> Dict[str, Any]:
        """
        Flux d'exécution de VerificationAgent :

        1. Pause de 3s pour respecter les rate limits Groq
        2. Lire et normaliser TOUTES les sorties précédentes
        3. Construire le prompt d'évaluation
        4. Appeler Groq
        5. Parser le JSON de rapport (avec fallback si parsing échoue)
        6. Retourner le rapport + tokens
        """

        # ── Étape 1 : Pause anti-rate-limit ──────────────────────────────
        # RAGAgent vient d'appeler Groq → on attend 3s pour éviter le 429.
        # Dans l'architecture distribuée, cette pause n'est pas nécessaire
        # car les agents s'exécutent en parallèle.
        time.sleep(3)

        query = state["user_query"]

        # ── Étape 2 : Normalisation des sorties précédentes ───────────────
        # state.get() peut retourner None si un agent précédent a échoué.
        # On remplace None par des chaînes descriptives pour que le LLM
        # comprenne ce qui s'est passé (plutôt que de voir "None").
        plan_text = state.get("plan")           or "Aucun plan disponible"
        rag_text  = state.get("retrieved_docs") or "Aucun document récupéré"
        tool_text = state.get("tool_results")   or "Aucun outil utilisé"

        # ── Étape 3 : Construction du prompt d'évaluation ─────────────────
        # Chaque section est tronquée pour rester dans la fenêtre de contexte.
        # plan[:500], rag[:800], tool[:500] → total ~1800 tokens de contexte
        # + réponse JSON ~200 tokens → bien en dessous de max_tokens=1024.
        prompt = f"""Question originale : {query}

Plan d'action :
{plan_text[:500]}

Résultats RAG :
{rag_text[:800]}

Résultats des outils :
{tool_text[:500]}

Vérifie la cohérence de ces informations et produis ton rapport JSON."""

        # ── Étape 4 : Appel Groq avec gestion d'erreur ───────────────────
        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
        except Exception as e:
            logger.error(f"[VerificationAgent] API error: {e}")
            # Rapport de secours : confiance neutre (0.5), PROCEED par défaut
            # → le pipeline continue avec une confiance "inconnue"
            # plutôt que de s'arrêter sur une erreur API
            return {
                "verification_report": {
                    "confidence_score": 0.5,
                    "consistency_check": "Partiel",
                    "potential_hallucinations": [],
                    "missing_information": [],
                    "quality_score": 0.5,
                    "recommendation": "PROCEED",
                    "verification_notes": f"Vérification impossible (erreur API): {str(e)[:100]}",
                },
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

        # ── Étape 5 : Parsing JSON robuste ───────────────────────────────
        raw = response.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                report = json.loads(match.group())
            except json.JSONDecodeError:
                report = None
        else:
            report = None

        # Si le parsing a échoué → rapport par défaut conservateur (0.7, PROCEED)
        if report is None:
            report = {
                "confidence_score": 0.7,
                "consistency_check": "Partiel",
                "potential_hallucinations": [],
                "missing_information": [],
                "quality_score": 0.7,
                "recommendation": "PROCEED",
                "verification_notes": "Format JSON non reçu — rapport auto-généré",
            }

        # ── Étape 6 : Extraction des tokens ──────────────────────────────
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            tu = response.response_metadata['token_usage']
            tokens_data = {
                "prompt_tokens":     tu.get('prompt_tokens', 0),
                "completion_tokens": tu.get('completion_tokens', 0),
                "total_tokens":      tu.get('total_tokens', 0)
            }
        else:
            tokens_data = {
                "prompt_tokens":     len(prompt) // 4,
                "completion_tokens": len(raw) // 4,
                "total_tokens":      (len(prompt) + len(raw)) // 4
            }

        logger.info(
            f"[VerificationAgent] Score: {report.get('confidence_score')} "
            f"| Rec: {report.get('recommendation')} "
            f"| Tokens: {tokens_data.get('total_tokens', 0)}"
        )

        return {"verification_report": report, "tokens": tokens_data}

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        """Expose le score de confiance et la recommandation dans les métriques."""
        report = state.get("verification_report") or {}
        return {
            "confidence_score": report.get("confidence_score", 0),
            "recommendation":   report.get("recommendation", "UNKNOWN"),
        }
```

---

## 4.5 Agent de synthèse (`SynthesisAgent`)

### Rôle et position dans le pipeline

`SynthesisAgent` est le **dernier agent**, le seul dont la sortie (`final_answer`) est directement renvoyée à l'utilisateur. Son rôle est d'agréger intelligemment toutes les sorties précédentes en une réponse Markdown cohérente, structurée et adaptée au niveau académique.

Il est le seul agent à avoir une vue complète de tout ce qui s'est passé dans le pipeline : plan, documents, outils, vérification. Sa tâche est essentiellement de **réconcilier et structurer** ces informations hétérogènes.

### Mode FALLBACK : quand la vérification recommande de tout ignorer

Si `VerificationAgent` retourne `"recommendation": "FALLBACK"`, cela signifie que les informations accumulées sont jugées trop peu fiables pour être utilisées. Dans ce cas, `SynthesisAgent` **ignore tout le contexte** et demande directement au LLM de répondre à la question originale depuis ses connaissances propres. C'est un mécanisme de dernière chance pour fournir quand même une réponse utile.

### Construction de la réponse par sections conditionnelles

Le contexte envoyé au LLM est construit dynamiquement : seules les sections non vides sont incluses. Si `ToolsAgent` n'a trouvé aucun outil pertinent, la section "Résultats des outils" n'est pas incluse. Cela évite d'envoyer du bruit au LLM final.

```python
# backend/agents/synthesis_agent.py — Code complet annoté

from typing import Dict, Any
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from backend.agents.base import BaseAgent
from backend.state import AcademicState

logger = logging.getLogger(__name__)

# Le prompt de synthèse est le plus riche en directives car c'est
# le LLM final qui parle directement à l'utilisateur.
# "Markdown" → réponse structurée avec titres, listes, code blocks
# "niveau académique" → vocabulaire précis, citations, rigueur
# "score de confiance bas" → transparence sur les incertitudes
SYSTEM_PROMPT = """Vous êtes un assistant IA conversationnel ultra-rapide avec une mémoire parfaite.

## 📋 RÈGLES D'OR (Priorité Absolue)

1. **MÉMOIRE INSTANTANÉE** : Si l'utilisateur s'est présenté (via user_name), saluez-le par son nom. Intégrez ses préférences dans le ton de la réponse.

2. **RÉPONSE RAPIDE** : Soyez direct et efficace. Si l'utilisateur demande son nom, répondez immédiatement en utilisant les données de l'état.

3. **CONTEXTE GLOBAL** : Utilisez l'historique complet et les analyses des experts (RAG, Outils) pour fournir une réponse unifiée.

## 🎓 DIRECTIVES ACADÉMIQUES

- Structurez vos réponses en Markdown (titres, listes).
- Restez rigoureux et citez systématiquement les sources documentaires.
- En cas d'incertitude (confiance < 0.6), mentionnez-le explicitement.
- Terminez par une conclusion synthétique et rigoureuse."""


class SynthesisAgent(BaseAgent):
    name = "synthesis"
    description = (
        "Dernier agent du pipeline : agrège toutes les sorties en une "
        "réponse Markdown structurée et adaptée au niveau académique."
    )

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """
        max_tokens=1024 : SynthesisAgent produit la réponse la plus longue
        du pipeline → on lui alloue le double des autres agents.
        C'est le seul agent où une réponse longue est souhaitable.
        """
        self.llm = ChatGroq(model=model, max_tokens=1024, temperature=0.1)

    def process(self, state: AcademicState) -> Dict[str, Any]:
        """
        Flux d'exécution de SynthesisAgent :

        1. Lire toutes les sorties des agents précédents
        2. Extraire le score de confiance et la recommandation de VerificationAgent
        3. Construire le contexte de synthèse par sections conditionnelles
        4. Adapter le prompt selon la recommandation (PROCEED vs FALLBACK)
        5. Appeler Groq
        6. Extraire les tokens
        7. Retourner final_answer
        """

        query       = state["user_query"]
        plan        = state.get("plan", "")
        rag_output  = state.get("retrieved_docs", "")
        tool_output = state.get("tool_results", "")

        # ── Lecture du rapport de vérification ───────────────────────────
        # "or {}" gère le cas où verification_report est None
        # (VerificationAgent a échoué ou n'a pas encore tourné)
        verification   = state.get("verification_report") or {}
        confidence     = verification.get("confidence_score", 0.8)
        recommendation = verification.get("recommendation", "PROCEED")
        notes          = verification.get("verification_notes", "")

        # ── Construction du contexte par sections conditionnelles ─────────
        # Chaque section n'est ajoutée que si elle contient du contenu.
        # L'ordre des sections = ordre naturel du raisonnement :
        # question → plan → recherche → outils → vérification → avertissement
        sections = [f"**Question :** {query}\n"]

        if plan:
            sections.append(f"**Plan d'analyse :**\n{plan}\n")

        if rag_output:
            sections.append(f"**Recherche documentaire :**\n{rag_output}\n")

        # Les résultats d'outils sont exclus s'ils contiennent le message
        # par défaut "Aucun outil externe requis" → pas de bruit inutile
        if tool_output and tool_output != "Aucun outil externe requis pour cette question.":
            sections.append(f"**Résultats des outils :**\n{tool_output}\n")

        if notes:
            sections.append(f"**Note de vérification :** {notes}\n")

        # Avertissement visible pour l'utilisateur si confiance < 60%
        if confidence < 0.6:
            sections.append(
                f"⚠️ **Avertissement :** Score de confiance bas — {confidence:.0%}. "
                "Certaines informations peuvent être incomplètes ou incertaines.\n"
            )

        # Séparateur "---" entre sections → le LLM voit des blocs distincts
        context = "\n---\n".join(sections)
        prompt  = (
            f"Synthétise les informations suivantes en une réponse académique complète.\n\n"
            f"{context}\n\nProduis maintenant la réponse finale structurée."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # ── Adaptation selon la recommandation de VerificationAgent ──────
        # FALLBACK : les informations sont trop peu fiables pour être utilisées.
        # On remplace le prompt complet par une question directe sans contexte.
        # Le LLM répond alors depuis ses connaissances propres uniquement.
        if recommendation == "FALLBACK":
            logger.warning("[SynthesisAgent] Mode FALLBACK activé — contexte ignoré")
            messages[1] = HumanMessage(
                content=f"Réponds directement à cette question académique : {query}"
            )

        # ── Appel Groq ────────────────────────────────────────────────────
        try:
            response = self.llm.invoke(messages)

            # Extraction des tokens réels (Groq retourne token_usage dans response_metadata)
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                tu = response.response_metadata['token_usage']
                tokens_data = {
                    "prompt_tokens":     tu.get('prompt_tokens', 0),
                    "completion_tokens": tu.get('completion_tokens', 0),
                    "total_tokens":      tu.get('total_tokens', 0)
                }
            else:
                tokens_data = {
                    "prompt_tokens":     len(prompt) // 4,
                    "completion_tokens": len(response.content) // 4,
                    "total_tokens":      (len(prompt) + len(response.content)) // 4
                }

            # final_answer est la clé lue par l'API FastAPI pour retourner
            # la réponse à l'utilisateur via /api/query
            return {"final_answer": response.content, "tokens": tokens_data}

        except Exception as e:
            logger.error(f"[SynthesisAgent] Erreur LLM: {e}")
            return {
                "final_answer": (
                    f"Une erreur est survenue lors de la synthèse finale. "
                    f"Détail technique : {str(e)[:100]}"
                ),
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        """Expose le score de confiance final utilisé pour la synthèse."""
        verification = state.get("verification_report") or {}
        return {
            "confidence_score": verification.get("confidence_score", 0.8),
            "recommendation":   verification.get("recommendation", "PROCEED"),
        }
```

---

## 4.6 Registre des agents et extensibilité (`AgentRegistry`)

### Pourquoi un registre ?

Sans registre, chaque fois qu'on ajoute un agent, il faudrait modifier l'orchestrateur, le routeur, l'API, les tests... Le registre **centralise** la connaissance des agents existants. L'orchestrateur ne connaît pas les agents directement — il interroge le registre au démarrage et construit le graphe dynamiquement.

### Pattern Singleton

`AgentRegistry` utilise le pattern **Singleton** : `__new__` vérifie si une instance existe déjà dans `_instance`. Si oui, elle la retourne sans en créer une nouvelle. Peu importe combien de fois on importe `registry` dans différents fichiers, c'est toujours le même objet en mémoire.

```python
# backend/agents/registry.py — Code complet annoté

from typing import Dict, List, Optional
import logging
from backend.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registre central de tous les agents — Pattern Singleton.

    SINGLETON : garantit qu'une seule instance existe dans toute l'application.
    Tous les modules qui font 'from backend.agents.registry import registry'
    reçoivent le même objet. Le registre est donc une source de vérité unique.

    DÉCOUPLAGE : l'orchestrateur, le routeur et l'API appellent registry.get()
    ou registry.list_agents() pour connaître les agents disponibles.
    Ils ne font jamais d'import direct des classes d'agents.
    → Ajouter un agent = l'enregistrer. Rien d'autre à changer.
    """
    _instance = None  # variable de classe (partagée entre toutes les instances)

    def __new__(cls):
        """
        Méthode appelée AVANT __init__ à chaque instanciation.
        Si _instance existe déjà → retourne l'existant (pas de nouvel objet).
        Si _instance est None → crée l'objet, initialise _agents, mémorise dans _instance.

        _agents est un dict {nom_agent: instance_agent}
        Ex: {"planning": PlanningAgent(), "rag": RAGAgent(), ...}
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents: Dict[str, BaseAgent] = {}
        return cls._instance

    def register(self, agent: BaseAgent) -> None:
        """
        Enregistre un agent dans le registre.

        Si un agent avec le même nom existe déjà, il est silencieusement
        remplacé. Utile pour mettre à jour un agent sans redémarrer.

        L'agent.name est utilisé comme clé du dict → doit être unique.
        Convention : snake_case court (ex: "planning", "rag", "tools").
        """
        self._agents[agent.name] = agent
        logger.info(f"[Registry] ✅ Agent enregistré: '{agent.name}' ({agent.__class__.__name__})")

    def unregister(self, name: str) -> None:
        """
        Supprime un agent du registre par son nom.

        Utile pour les tests (retirer un agent mock) ou
        pour désactiver dynamiquement un agent sans redémarrer le serveur.
        L'opération est silencieuse si l'agent n'existe pas.
        """
        if name in self._agents:
            del self._agents[name]
            logger.info(f"[Registry] ❌ Agent supprimé: '{name}'")

    def get(self, name: str) -> Optional[BaseAgent]:
        """
        Récupère une instance d'agent par son nom.

        Retourne None si l'agent n'est pas enregistré (pas d'exception).
        L'appelant doit vérifier si le retour est None avant utilisation.

        Usage typique :
        rag = registry.get("rag")
        if rag: rag.add_document(...)
        """
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """Retourne la liste des noms d'agents enregistrés."""
        return list(self._agents.keys())

    def all_agents(self) -> Dict[str, BaseAgent]:
        """Retourne une copie du dict complet {nom: instance}."""
        return dict(self._agents)

    def agent_descriptions(self) -> Dict[str, str]:
        """
        Retourne {nom: description} pour tous les agents enregistrés.
        Utilisé par l'endpoint GET /api/agents et par le SYSTEM_PROMPT
        de ToolsAgent pour construire dynamiquement la liste des outils.
        """
        return {name: agent.description for name, agent in self._agents.items()}


# ── Instance globale ───────────────────────────────────────────────────
# Importée dans tous les modules qui ont besoin d'accéder aux agents.
# from backend.agents.registry import registry
registry = AgentRegistry()
```

### Ajouter un agent en 3 étapes

```python
# ── ÉTAPE 1 : Créer le fichier de l'agent ─────────────────────────────
# backend/agents/citation_agent.py

class CitationAgent(BaseAgent):
    """
    Agent de citation académique — exemple d'extension minimale.

    Illustre le pattern d'extension :
    - Hérite de BaseAgent (obligatoire)
    - Définit name et description (obligatoire)
    - Implémente process() (obligatoire)
    - Lit les sorties d'autres agents via l'état partagé (A2A)
    - Ne modifie aucun autre fichier du projet (découplage total)
    """
    name = "citation"
    description = "Génère des citations APA/MLA/IEEE depuis les sources RAG."

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.llm = ChatGroq(model=model, max_tokens=512)

    def process(self, state: AcademicState) -> Dict[str, Any]:
        # A2A : lecture des documents récupérés par RAGAgent
        # Si RAGAgent n'a rien trouvé, on retourne un message neutre
        rag_docs = state.get("retrieved_docs", "")
        query    = state["user_query"]

        if not rag_docs or "aucun document" in rag_docs.lower():
            existing = state.get("tool_results", "") or ""
            return {"tool_results": existing + "\n\n📖 **Citations :** Aucune source à citer."}

        # Génération des citations APA depuis les documents trouvés
        prompt = (
            f"À partir de ces informations sur '{query}', "
            f"génère 2-3 citations académiques au format APA :\n{rag_docs[:600]}"
        )
        response = self.llm.invoke([
            SystemMessage(content="Tu es un expert en citations académiques. Génère des citations APA précises."),
            HumanMessage(content=prompt),
        ])

        # On AJOUTE au tool_results existant (ne pas écraser les résultats de ToolsAgent)
        existing = state.get("tool_results", "") or ""
        return {"tool_results": existing + f"\n\n📖 **Citations APA :**\n{response.content}"}

    def get_metadata(self, state: AcademicState) -> Dict[str, Any]:
        return {"citation_style": "APA"}


# ── ÉTAPE 2 : Enregistrer dans main.py ────────────────────────────────
# Ajouter ces 3 lignes après les autres registrations dans backend/main.py :

# from backend.agents.citation_agent import CitationAgent
# registry.register(CitationAgent())
# orchestrator.rebuild_graph()  # reconstruit le graphe LangGraph avec le nouvel agent


# ── ÉTAPE 3 : Vérifier l'enregistrement ───────────────────────────────
# curl http://localhost:8000/api/agents
# → {"agents": ["planning", "rag", "tools", "verification", "synthesis", "citation"]}
```


## **Ressources du projet**

Afin de faciliter la compréhension, la reproductibilité et la poursuite de la lecture de ce projet de recherche, l’ensemble des ressources utilisées est mis à disposition ci-dessous.

### **Questions d’étude (160 questions brutes par architecture)**

[Accéder aux questions d’étude](https://drive.google.com/file/d/1KxcRF8VK9NqW_yjPUW-WgKlcsN5eL6b4/view)

---

### **Dataset – Architecture hiérarchique (résultats annotés)**

[Accéder au dataset hiérarchique](https://drive.google.com/file/d/1dcOwou6JVUA68kl5kPCj0jiz2jEOUPop/view)

---

### **Dataset – Architecture distribuée (résultats annotés)**

[Accéder au dataset distribué](https://drive.google.com/file/d/1HHVlSkyogRWjRE2g1GrIuNCG4xcSZ1sb/view)

---

### **Notebook d’expérimentation**

(Prétraitement, entraînement, évaluation et étude d’ablation)

[Ouvrir le notebook d’expérimentation](https://drive.google.com/file/d/1FDWvlUyVW47MFLkkxf3gtsI1Q7Rd7Zs3/view)

---

### **Meilleur modèle retenu (pipeline sérialisé – Joblib)**

[Télécharger le modèle joblib](https://drive.google.com/file/d/1WbaPRPV0YPI0Ex_daTexzFJF0g5arV27/view)

---

## **Dépôt GitHub officiel**

Le code source complet du projet est disponible sur le dépôt GitHub officiel suivant :

[Accéder au dépôt GitHub](https://github.com/hinimdoumorsia/MultiAgentStudyArchitecture)