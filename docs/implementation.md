# 8. Implémentation

## 8.1 Stack technique

| Composant | Technologie | Rôle |
| :--- | :--- | :--- |
| **Langage** | Python 3.10+ | Langage principal |
| **Orchestration** | LangGraph (`StateGraph`) | Graphe d'exécution hiérarchique |
| **API** | FastAPI + Uvicorn | Serveur HTTP REST |
| **LLMs** | Claude 3.5 Haiku (`claude-3-5-haiku-20241022`) | Agents planning, RAG, tools, verification |
| **LLMs** | Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`) | Agent synthesis (plus puissant) |
| **Base vectorielle** | ChromaDB (persistant, `./data/chroma_db`) | Retrieval-Augmented Generation (RAG) |
| **Base relationnelle** | SQLite (`./data/memory.db`) | Historique, métriques, données Meta-Router |
| **Protocole outils** | MCP — JSON-RPC 2.0 sur stdio | Interface standardisée des outils externes |
| **Bus d'événements** | `EventBus` (threading, singleton) | Communication P2P inter-agents |
| **ML / Meta-Router** | scikit-learn, TF-IDF, SMOTE | Classification supervisée architecture |
| **Sérialisation modèle** | joblib | Pipeline ML persisté sur disque |

---

## 8.2 Structure du projet

```
academic-mas/
│
├── backend/
│   ├── main.py                      # Serveur FastAPI — endpoints REST
│   ├── orchestrator.py              # LangGraph StateGraph (architecture hiérarchique)
│   ├── state.py                     # AcademicState + RouterDecision (état partagé)
│   │
│   ├── agents/
│   │   ├── base.py                  # BaseAgent (classe abstraite)
│   │   ├── registry.py              # AgentRegistry (scalabilité dynamique)
│   │   ├── planning_agent.py        # Agent 1 : Planification
│   │   ├── rag_agent.py             # Agent 2 : Retrieval / RAG
│   │   ├── tools_agent.py           # Agent 3 : Outils / MCP
│   │   ├── verification_agent.py    # Agent 4 : Vérification qualité
│   │   ├── synthesis_agent.py       # Agent 5 : Synthèse finale
│   │   └── example_custom_agent.py  # Template pour nouveaux agents
│   │
│   ├── distributed/                 # Architecture P2P
│   │   ├── event_bus.py             # Bus d'événements thread-safe (singleton)
│   │   ├── distributed_agents.py    # Wrappers P2P des agents existants
│   │   └── p2p_runner.py            # Point d'entrée du pipeline P2P
│   │
│   ├── meta_router/
│   │   └── meta_router.py           # Sélection dynamique hiérarchique vs P2P
│   │
│   ├── memory/
│   │   └── memory_manager.py        # SessionMemory (RAM) + PersistentMemory (SQLite)
│   │
│   ├── mcp/
│   │   ├── __init__.py              # Export : MCPServer, mcp_server
│   │   └── server.py                # Vrai serveur MCP JSON-RPC 2.0
│   │
│   └── utils/
│       └── router.py                # Routeur interne (hiérarchique) — regex + heuristiques
│
├── data/                            # Créé automatiquement au premier lancement
│   ├── chroma_db/                   # Base vectorielle RAG (ChromaDB persistant)
│   └── memory.db                    # Historique SQLite — données de recherche
│
├── data_collection/                 # Scripts de collecte du dataset expérimental
│   ├── collector.py                 # Collecte simultanée sur les deux architectures
│   ├── collector_hier.py            # Collecte architecture hiérarchique uniquement
│   └── collector_dist.py            # Collecte architecture distribuée uniquement
│
├── image/                           # Images pour la documentation
├── models/                          # Notebook d'expérimentation + meilleur modèle (joblib)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 8.3 État partagé — `AcademicState` et `RouterDecision`

L'état partagé est le **contrat de communication** entre tous les composants du système. Défini dans `backend/state.py`, il est utilisé par les deux architectures.

### `AcademicState`

```python
class AcademicState(TypedDict):
    # ── Conversation ───────────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]  # Historique LangGraph
    user_query: str                                        # Question brute
    session_id: str                                        # Identifiant de session
    user_name: Optional[str]                               # Stockage immédiat du nom (Règle d'Or #1)

    # ── Résultats des agents ───────────────────────────────────────────
    router_decision: Optional[RouterDecision]  # Décision du routeur interne
    plan: Optional[str]                        # PlanningAgent → plan structuré
    retrieved_docs: Optional[str]              # RAGAgent → passages extraits
    tool_results: str                          # ToolsAgent → résultats outils
    verification_report: Optional[Dict]        # VerificationAgent → rapport qualité
    final_answer: Optional[str]                # SynthesisAgent → réponse finale

    # ── Métriques (collectées par run) ────────────────────────────────
    agent_results: Annotated[List[AgentResult], operator.add]
    total_latency_ms: float
    run_id: str

    # ── Gestion des erreurs ────────────────────────────────────────────
    errors: Annotated[List[str], operator.add]
    retry_count: int
    iteration_count: Annotated[int, operator.add]
```

### `RouterDecision`

```python
class RouterDecision(TypedDict):
    selected_agents: List[str]       # Ex: ["planning", "rag", "verification", "synthesis"]
    reasoning: str                   # Justification textuelle du routage
    estimated_complexity: str        # "low" | "medium" | "high"
    context_load: float              # Charge contextuelle normalisée [0.0, 1.0]
    estimated_cost: Dict[str, Any]   # {tokens, time_ms, monetary_cents, agents_needed}
```

### `AgentResult`

Chaque agent produit un `AgentResult` ajouté à `state["agent_results"]` via `operator.add` (accumulation automatique LangGraph) :

```python
class AgentResult(TypedDict):
    agent_name: str      # Nom de l'agent ("planning", "rag", etc.)
    output: str          # Sortie textuelle de l'agent
    confidence: float    # Score de confiance [0.0, 1.0]
    latency_ms: float    # Temps d'exécution en millisecondes
    success: bool        # Succès ou échec de l'exécution
    metadata: Dict       # Données libres (tokens, outils utilisés, etc.)
```

---

## 8.4 Orchestrateur hiérarchique — `AcademicOrchestrator`

### Principe de construction dynamique du graphe

L'orchestrateur (`backend/orchestrator.py`) construit le graphe LangGraph **de façon dynamique** à partir du registre d'agents. L'ajout ou la suppression d'un agent ne nécessite pas de modifier le code de l'orchestrateur — le graphe se reconstruit automatiquement.

```python
# Structure fixe du graphe compilé :
router → planning → rag → tools → verification → synthesis → END
```

**Point clé :** la structure est séquentielle et déterministe, sans boucle de vérification itérative (fixée à 1 passage unique). Cela garantit la prévisibilité et évite les boucles infinies.

### Construction du graphe

```python
def build_graph() -> StateGraph:
    graph = StateGraph(AcademicState)

    # Nœud d'entrée : routage
    graph.add_node("router", _router_node)

    # Nœuds agents — ajoutés dynamiquement depuis le registry
    for name, agent in registry.all_agents().items():
        graph.add_node(name, agent)

    graph.set_entry_point("router")

    # Arêtes séquentielles conditionnelles selon les agents enregistrés
    # router → planning → rag → tools → verification → synthesis → END
    ...
    return graph.compile()
```

### Nœud routeur interne

```python
def _router_node(state: AcademicState) -> Dict[str, Any]:
    # 1. Récupérer le contexte de session (RAM ou SQLite)
    session_ctx = memory_manager.get_session_context(state["session_id"])

    # 2. Enrichir la requête avec le contexte si disponible
    query = state["user_query"]
    if session_ctx:
        query = f"[Contexte session]\n{session_ctx}\n\n[Question actuelle]\n{query}"

    # 3. Appliquer les heuristiques de routage
    decision = select_agents({**state, "user_query": query})
    return {"router_decision": decision}
```

### Cycle de vie d'une requête

```
AcademicOrchestrator.run(query, session_id)
        │
        ▼
Initialisation de l'AcademicState (toutes les clés à None / "")
        │
        ▼
graph.invoke(initial_state)
        │
        ├── router_node     → RouterDecision (agents sélectionnés, complexité, coût estimé)
        ├── planning_node   → state["plan"]
        ├── rag_node        → state["retrieved_docs"]
        ├── tools_node      → state["tool_results"]
        ├── verification_node → state["verification_report"]
        └── synthesis_node  → state["final_answer"]
        │
        ▼
memory_manager.record(session_id, run_id, query, answer, agents, confidence, latency)
        │
        ▼
Retour du payload JSON complet
```

### Interface publique

```python
from backend.orchestrator import AcademicOrchestrator

orchestrator = AcademicOrchestrator()

# Lancer une requête
result = orchestrator.run(
    query="Explique le théorème de Bayes avec un exemple numérique.",
    session_id="session-xyz"
)

# Recharger le graphe après ajout d'un agent (hot-reload)
orchestrator.rebuild_graph()

# Lister les agents enregistrés
agents = orchestrator.get_registered_agents()
# → {"planning": "Analyse la question...", "rag": "Recherche documentaire...", ...}
```

---

## 8.5 Scalabilité — Ajouter ou retirer un agent

### Ajouter un agent (3 étapes)

**Étape 1 — Créer le fichier agent**

```python
# backend/agents/my_agent.py
from backend.agents.base import BaseAgent
from backend.state import AcademicState

class MyAgent(BaseAgent):
    name = "my_agent"
    description = "Description de ce que fait cet agent."

    def process(self, state: AcademicState) -> dict:
        # Lire depuis l'état
        plan = state.get("plan", "")
        # Produire un résultat
        return {"tool_results": f"Résultat de MyAgent basé sur : {plan}"}
```

**Étape 2 — Enregistrer dans le registry (architecture hiérarchique)**

```python
from backend.agents.my_agent import MyAgent
from backend.agents.registry import registry

registry.register(MyAgent())
orchestrator.rebuild_graph()   # Hot-reload du graphe
```

**Étape 3 — Créer le wrapper distribué (architecture P2P)**

```python
# backend/distributed/distributed_agents.py
class DistributedMyAgent(DistributedAgentWrapper):
    trigger_events = [EventType.PLAN_CREATED]
    output_event   = EventType.TOOL_EXECUTED   # ou un nouvel EventType

    def _extract_payload(self, result, state):
        return {"tool_results": result.get("tool_results", "")}
```

### Retirer un agent

```python
# Architecture hiérarchique
registry.unregister("my_agent")
orchestrator.rebuild_graph()

# Architecture P2P : supprimer le wrapper du PeerToPeerRunner
```

---

## 8.6 Gestion des erreurs et reprise

### Architecture hiérarchique

Les erreurs sont capturées au niveau de l'orchestrateur et accumulées dans `state["errors"]` via `operator.add`. En cas d'exception critique, le graphe retourne un état de fallback :

```python
try:
    final_state = self._graph.invoke(initial_state)
except Exception as e:
    final_state = {
        **initial_state,
        "errors": [str(e)],
        "final_answer": f"Erreur système : {e}"
    }
```

L'agent de vérification produit une recommandation `FALLBACK` lorsque le `confidence_score < 0.6`, déclenchant un mode de réponse dégradé dans le `SynthesisAgent`.

```json
{
  "confidence_score": 0.45,
  "recommendation": "FALLBACK",
  "consistency_check": "Incohérent"
}
```

### Architecture distribuée (P2P)

En mode P2P, chaque agent défaillant publie un événement `ERROR` sur le bus, ce qui débloque le `threading.Event` du `PeerToPeerRunner` sans arrêter les autres agents en cours d'exécution :

```python
bus.publish(Event(
    type=EventType.ERROR,
    payload={"error": str(e), "source": self.name},
    source=self.name,
    session_id=session_id,
))
```

Un **timeout de 180 secondes** est configuré sur le `PeerToPeerRunner`. Au-delà, le pipeline retourne l'état partiel disponible.

### Gestion du rate limit API Anthropic

Le `DistributedAgentWrapper` intègre un mécanisme de **throttling** pour respecter les limites de débit de l'API Anthropic. Un délai minimum est imposé entre deux appels LLM pour la même session, évitant les erreurs `429 Too Many Requests` lorsque plusieurs agents se déclenchent en parallèle après `PLAN_CREATED`.

---

## 8.7 Endpoints FastAPI

Le serveur FastAPI (`backend/main.py`) expose les endpoints suivants :

| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/query` | Soumettre une requête (hiérarchique, p2p, ou compare) |
| `POST` | `/api/documents` | Ajouter un document à la base RAG (ChromaDB) |
| `GET` | `/api/stats` | Statistiques comparatives par architecture |
| `GET` | `/api/tools/manifest` | Manifeste MCP des outils disponibles |
| `GET` | `/docs` | Documentation interactive Swagger UI |

### Exemple — Soumettre une requête

```bash
# Architecture hiérarchique
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explique le théorème de Bayes avec un exemple.",
    "architecture": "hierarchical",
    "session_id": "ma-session-123"
  }'

# Architecture distribuée P2P
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explique le théorème de Bayes avec un exemple.",
    "architecture": "p2p",
    "session_id": "ma-session-123"
  }'

# Mode comparaison — les deux en parallèle
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explique le théorème de Bayes avec un exemple.",
    "architecture": "compare"
  }'
```

### Exemple — Ajouter un document à la base RAG

```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Le théorème de Bayes stipule que P(A|B) = P(B|A) * P(A) / P(B)...",
    "source": "Cours de probabilités, Chapitre 3"
  }'
```

### Payload de retour `/api/query`

```json
{
  "session_id": "ma-session-123",
  "run_id": "f1d48507-8f19-4911-9dc3-e27302e11afd",
  "query": "Explique le théorème de Bayes...",
  "final_answer": "**Réponse finale**...",
  "plan": "**Stratégie :** ...",
  "retrieved_docs": "Documents extraits de ChromaDB...",
  "tool_results": "Aucun outil externe requis.",
  "verification_report": {
    "confidence_score": 0.87,
    "consistency_check": "Cohérent",
    "potential_hallucinations": [],
    "recommendation": "PROCEED",
    "quality_score": 0.85
  },
  "router_decision": {
    "selected_agents": ["planning", "rag", "verification", "synthesis"],
    "reasoning": "planning systématique | recherche documentaire détectée",
    "estimated_complexity": "medium",
    "context_load": 0.52,
    "estimated_cost": {
      "tokens": 1200,
      "time_ms": 3200,
      "agents_needed": 4
    }
  },
  "agent_results": [
    {"agent_name": "planning", "latency_ms": 452.62, "confidence": 0.85, "success": true},
    {"agent_name": "rag",      "latency_ms": 2199.81, "confidence": 0.85, "success": true},
    {"agent_name": "verification", "latency_ms": 3501.19, "confidence": 0.80, "success": true},
    {"agent_name": "synthesis",    "latency_ms": 4874.47, "confidence": 0.85, "success": true}
  ],
  "total_latency_ms": 11028.1,
  "errors": []
}
```