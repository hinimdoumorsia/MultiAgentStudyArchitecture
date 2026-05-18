# AcademiMAS — Orchestrateur Multi-Agents pour l'Assistance Académique

> Architecture hiérarchique multi-agents construite avec **LangGraph**, **Claude (Anthropic)**, **ChromaDB** et **FastAPI**, avec une interface React moderne.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture hiérarchique](#architecture-hiérarchique)
3. [Les agents et leurs rôles](#les-agents-et-leurs-rôles)
4. [Communication entre agents (A2A)](#communication-entre-agents-a2a)
5. [Mémoire : session et persistante](#mémoire--session-et-persistante)
6. [Serveur MCP](#serveur-mcp)
7. [Routeur dynamique](#routeur-dynamique)
8. [Scalabilité — ajouter / retirer un agent](#scalabilité--ajouter--retirer-un-agent)
9. [Installation et lancement](#installation-et-lancement)
10. [Structure du projet](#structure-du-projet)
11. [Métriques et évaluation](#métriques-et-évaluation)

---

## Vue d'ensemble

AcademiMAS est un système multi-agents capable de répondre à des questions académiques complexes en décomposant le travail entre plusieurs agents spécialisés coordonnés par un orchestrateur central.

```
Utilisateur → Orchestrateur → Routeur dynamique → Agents spécialisés → Synthèse → Réponse
```

**Capacités** :
- Décomposition automatique de questions complexes
- Recherche vectorielle dans une base documentaire (RAG)
- Exécution d'outils : calculs, code Python, Wikipedia
- Vérification de cohérence et détection d'hallucinations
- Mémoire de session + persistance inter-sessions
- Interface de chat moderne avec visualisation du pipeline

---

## Architecture hiérarchique

Le système s'organise en **3 couches** :

```
┌─────────────────────────────────────────────────────────────┐
│  COUCHE 0 — Interface utilisateur (React)                    │
└────────────────────────┬────────────────────────────────────┘
                         │ requête HTTP
┌────────────────────────▼────────────────────────────────────┐
│  COUCHE 1 — Orchestrateur (LangGraph StateGraph)             │
│                                                              │
│   ┌─────────────────┐    ┌──────────────────────────────┐   │
│   │ Orchestrateur   │    │ Routeur dynamique            │   │
│   │ central         │───▶│ Score confiance · Coût       │   │
│   │ Coordination    │    │ Type requête · Charge ctx.   │   │
│   └─────────────────┘    └──────────────────────────────┘   │
│                                                              │
│   ┌─────────────┐          ┌──────────────────────────┐     │
│   │ Mémoire     │          │ Serveur MCP              │     │
│   │ de session  │          │ (outils externes)        │     │
│   └─────────────┘          └──────────────────────────┘     │
│   ┌─────────────┐                                            │
│   │ Mémoire     │                                            │
│   │ persistante │                                            │
│   └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  COUCHE 2 — Agents spécialisés                               │
│                                                              │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Planning   │ │ RAG        │ │ Tools    │ │Verification│  │
│  │ Agent      │ │ Agent      │ │ Agent    │ │ Agent      │  │
│  └────────────┘ └────────────┘ └──────────┘ └────────────┘  │
│        ←─────────── Protocole A2A (état partagé) ──────────→ │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  COUCHE 3 — Synthèse finale                                  │
│                                                              │
│                  ┌──────────────┐                            │
│                  │ Synthesis    │                            │
│                  │ Agent        │                            │
│                  └──────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

**Pourquoi hiérarchique et non peer-to-peer ?**

| Critère | Hiérarchique ✓ | Peer-to-peer |
|---|---|---|
| Traçabilité | Flux centralisé, chaque décision loggée | Difficile à suivre |
| Cohérence | L'orchestrateur garantit l'ordre | Risque de conflits |
| Reprise sur erreur | Point de contrôle unique | Détection complexe |
| Scalabilité | Ajout d'agent sans refactoring | Recâblage de tous les nœuds |
| Coût | Routeur évite les appels inutiles | Tous les agents sollicités |

---

## Les agents et leurs rôles

### 1. PlanningAgent (`planning`)

**Rôle** : Point d'entrée systématique. Analyse la question et produit un plan d'action structuré.

**Ce qu'il fait** :
- Décompose la question en sous-tâches ordonnées
- Estime la complexité : `low` / `medium` / `high`
- Détermine si RAG et/ou outils sont nécessaires
- Influence le routeur via l'état partagé

**Modèle** : `claude-3-5-haiku-20241022` (rapide, économique)

**Sortie** : `state["plan"]` — texte structuré Markdown

**Exemple de sortie** :
```
Stratégie : Recherche + calcul combinés
Complexité : medium
Étapes :
  1. Rechercher la définition du théorème de Bayes
  2. Trouver un exemple concret dans les documents
  3. Calculer un exemple numérique
```

---

### 2. RAGAgent (`rag`)

**Rôle** : Retrieval-Augmented Generation — recherche dans la base documentaire académique.

**Ce qu'il fait** :
- Requête vectorielle dans ChromaDB (cosine similarity)
- Enrichit la requête avec le plan du PlanningAgent (A2A)
- Retourne les passages les plus pertinents avec scores de similarité
- Fallback sur les connaissances générales si aucun document indexé

**Modèle** : `claude-3-5-haiku-20241022`

**Stockage** : ChromaDB persistant (`./data/chroma_db`)

**Sortie** : `state["retrieved_docs"]` — passages + sources + scores

**Ajouter un document à la base** :
```python
from backend.agents.registry import registry
rag = registry.get("rag")
rag.add_document(content="...", source="Mon livre, p.42")
```

---

### 3. ToolsAgent (`tools`)

**Rôle** : Exécution d'outils externes — calcul, code, recherche web.

**Ce qu'il fait** :
- Décide automatiquement quels outils utiliser (via LLM)
- Exécute les outils de façon sécurisée (sandbox Python)
- Intègre les outils MCP enregistrés dynamiquement

**Outils disponibles** :

| Outil | Description | Exemple |
|---|---|---|
| `calculator` | Expressions mathématiques | `"math.sqrt(144)"` |
| `python_executor` | Code Python sandboxé | `"print(sum(range(10)))"` |
| `wikipedia_search` | Résumés Wikipedia FR/EN | `"mécanique quantique"` |

**Ajouter un outil dynamiquement** :
```python
from backend.agents.registry import registry
tools_agent = registry.get("tools")
tools_agent.register_tool("my_tool", my_function)
```

**Modèle** : `claude-3-5-haiku-20241022`

**Sortie** : `state["tool_results"]`

---

### 4. VerificationAgent (`verification`)

**Rôle** : Gardien de la qualité. Lit toutes les sorties des agents précédents et évalue leur cohérence.

**Ce qu'il fait** :
- Détecte les incohérences entre RAG et outils
- Identifie les affirmations potentiellement halluccinées
- Produit un score de confiance [0, 1]
- Émet une recommandation : `PROCEED` / `RETRY` / `FALLBACK`

**Modèle** : `claude-3-5-haiku-20241022`

**Sortie** : `state["verification_report"]` — JSON structuré :
```json
{
  "confidence_score": 0.87,
  "consistency_check": "Cohérent",
  "potential_hallucinations": [],
  "recommendation": "PROCEED",
  "quality_score": 0.85,
  "verification_notes": "Sources cohérentes, calcul vérifié."
}
```

---

### 5. SynthesisAgent (`synthesis`)

**Rôle** : Agrégateur final. Combine toutes les sorties en une réponse académique structurée.

**Ce qu'il fait** :
- Intègre plan + docs RAG + résultats outils + rapport de vérification
- Structure la réponse avec Markdown (titres, listes, code)
- Adapte le ton académique (précis, rigoureux, pédagogique)
- Signale les incertitudes si confidence < 60%
- Bascule en mode FALLBACK si la vérification le recommande

**Modèle** : `claude-3-5-sonnet-20241022` (plus puissant pour la synthèse)

**Sortie** : `state["final_answer"]` — réponse Markdown finale

---

## Communication entre agents (A2A)

Les agents communiquent via l'**état partagé LangGraph** (`AcademicState`). C'est le cœur du protocole Agent-to-Agent (A2A) dans ce système.

### Flux de données

```
PlanningAgent → state["plan"] → RAGAgent (enrichit sa requête)
                              → VerificationAgent (contexte)
                              → SynthesisAgent (structure)

RAGAgent      → state["retrieved_docs"] → VerificationAgent
                                        → SynthesisAgent

ToolsAgent    → state["tool_results"]   → VerificationAgent
                                        → SynthesisAgent

VerificationAgent → state["verification_report"] → SynthesisAgent
```

### Pourquoi l'état partagé ?

1. **Pas de couplage direct** : les agents ne se connaissent pas, ils lisent/écrivent l'état
2. **Traçabilité complète** : chaque modification est loggée avec l'agent responsable
3. **Reprise sur erreur** : si un agent échoue, l'état reste cohérent pour les suivants
4. **Thread-safe** : LangGraph gère la concurrence via `Annotated` + `operator.add`

### Structure de l'état

```python
class AcademicState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # historique
    user_query: str                   # question originale
    session_id: str                   # identifiant de session
    router_decision: RouterDecision   # décision du routeur
    plan: str                         # sortie PlanningAgent
    retrieved_docs: str               # sortie RAGAgent
    tool_results: str                 # sortie ToolsAgent
    verification_report: Dict         # sortie VerificationAgent
    final_answer: str                 # sortie SynthesisAgent
    agent_results: List[AgentResult]  # métriques par agent
    errors: List[str]                 # erreurs accumulées
```

---

## Mémoire : session et persistante

### Mémoire de session (`SessionMemory`)

- **Type** : In-memory Python dict
- **Durée de vie** : durée du processus serveur
- **Usage** : contexte des derniers tours de conversation
- **Capacité** : 20 derniers tours par session
- **Rôle dans l'architecture** : enrichit la requête envoyée au routeur pour maintenir la cohérence conversationnelle

### Mémoire persistante (`PersistentMemory`)

- **Type** : SQLite (`./data/memory.db`)
- **Durée de vie** : permanente (survit aux redémarrages)
- **Usage** : historique complet, statistiques, analytics
- **Schéma** :

```sql
CREATE TABLE conversations (
    id           INTEGER PRIMARY KEY,
    session_id   TEXT,
    run_id       TEXT,
    query        TEXT,
    answer       TEXT,
    agents_used  TEXT,   -- JSON array
    confidence   REAL,
    latency_ms   REAL,
    timestamp    TEXT,
    metadata     TEXT    -- JSON object
);
```

**Accès** :
```python
from backend.memory.memory_manager import memory_manager

# Récupérer l'historique d'une session
history = memory_manager.persistent.get_session_history("session-id", limit=10)

# Statistiques globales
stats = memory_manager.get_stats()
# → {"total_conversations": 42, "avg_confidence": 0.84, ...}
```

---

## Serveur MCP

Le module `backend/mcp/__init__.py` implémente un serveur **Model Context Protocol** léger.

**Outils MCP pré-enregistrés** :
- `latex_formatter` : formate une expression en LaTeX
- `citation_formatter` : génère une citation APA

**Ajouter un outil MCP** :
```python
from backend.mcp import mcp_server

mcp_server.register_tool(
    name="my_tool",
    fn=lambda input: f"résultat: {input}",
    description="Description de l'outil",
    input_schema={"type": "string"},
)
```

**Manifest MCP** (endpoint `/api/tools/manifest`) :
```json
{
  "tools": [
    {"name": "latex_formatter", "description": "...", "inputSchema": {...}},
    {"name": "citation_formatter", "description": "...", "inputSchema": {...}}
  ],
  "version": "1.0",
  "protocol": "MCP/1.0"
}
```

---

## Routeur dynamique

Le routeur (`backend/utils/router.py`) sélectionne les agents à activer pour chaque requête.

### Signaux de routage

| Signal | Source | Impact |
|---|---|---|
| **Type de requête** | Patterns regex sur la question | Active RAG, Tools, ou les deux |
| **Charge contextuelle** | Nombre de messages + longueur query | Évite la surcharge |
| **Agents disponibles** | Registry en temps réel | Routing adaptatif |
| **Complexité** | Nombre d'agents sélectionnés | Estimation coût |

### Logique de décision

```
Question → contient mots de recherche ? → active RAG
         → contient calcul/code ?        → active Tools
         → plusieurs agents actifs ?     → active Verification
         → toujours                      → Planning + Synthesis
```

### RouterDecision

```python
{
    "selected_agents": ["planning", "rag", "tools", "verification", "synthesis"],
    "reasoning": "recherche documentaire | calcul détecté | vérification multi-sources",
    "estimated_complexity": "high",
    "context_load": 0.12
}
```

---

## Scalabilité — ajouter / retirer un agent

### ✅ Ajouter un agent (3 étapes)

**Étape 1** : Créer le fichier agent

```python
# backend/agents/my_agent.py
from backend.agents.base import BaseAgent
from backend.state import AcademicState

class MyAgent(BaseAgent):
    name = "my_agent"
    description = "Ce que fait mon agent."

    def process(self, state: AcademicState) -> dict:
        # Lire l'état : state["user_query"], state["plan"], etc.
        # Retourner une mise à jour partielle de l'état
        return {"tool_results": "Mon résultat"}
```

**Étape 2** : Enregistrer dans le registry

```python
# Dans backend/main.py, à l'initialisation :
from backend.agents.my_agent import MyAgent
from backend.agents.registry import registry

registry.register(MyAgent())
orchestrator.rebuild_graph()  # hot-reload du graphe
```

**Étape 3** : (Optionnel) Ajouter des patterns au routeur

```python
# Dans backend/utils/router.py :
MY_PATTERNS = re.compile(r'\b(mon|pattern|spécifique)\b', re.I)

# Dans select_agents() :
if "my_agent" in available and MY_PATTERNS.search(query):
    selected.append("my_agent")
```

**C'est tout.** Aucun autre fichier à modifier.

---

### ❌ Retirer un agent (1 étape)

```python
from backend.agents.registry import registry
registry.unregister("my_agent")
orchestrator.rebuild_graph()
```

Le graphe se reconstruit sans cet agent. Les autres agents continuent de fonctionner.

---

### Pourquoi c'est scalable ?

1. **Registry singleton** : source de vérité unique pour tous les agents
2. **BaseAgent** : contrat commun — chaque agent est interchangeable
3. **État typé** : `AcademicState` découple les agents (pas d'appels directs)
4. **`rebuild_graph()`** : hot-reload sans redémarrer le serveur
5. **Routeur dynamique** : s'adapte automatiquement aux agents disponibles

---

## Installation et lancement

### Prérequis

- Python 3.10+
- Node.js 18+
- Clé API Anthropic

### Backend

```bash
# 1. Cloner et configurer
cd academic-mas
cp .env.example .env
# Éditer .env : ajouter ANTHROPIC_API_KEY=sk-ant-...

# 2. Installer les dépendances Python
pip install -r requirements.txt

# 3. Lancer le serveur FastAPI
python -m uvicorn backend.main:app --reload --port 8000
```

Le serveur sera disponible sur `http://localhost:8000`.
Documentation API interactive : `http://localhost:8000/docs`

### Frontend

```bash
# Dans un second terminal
cd frontend
npm install
npm run dev
```

L'interface sera disponible sur `http://localhost:3000`.

### Ajouter des documents à la base RAG

```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Le théorème de Bayes stipule que P(A|B) = P(B|A) * P(A) / P(B)...",
    "source": "Cours de probabilités, Chapitre 3"
  }'
```

---

## Structure du projet

```
academic-mas/
│
├── backend/
│   ├── main.py                      # Serveur FastAPI
│   ├── orchestrator.py              # LangGraph StateGraph
│   ├── state.py                     # AcademicState (état partagé)
│   │
│   ├── agents/
│   │   ├── base.py                  # BaseAgent (classe abstraite)
│   │   ├── registry.py              # AgentRegistry (scalabilité)
│   │   ├── planning_agent.py        # Agent 1 : Planification
│   │   ├── rag_agent.py             # Agent 2 : Retrieval / RAG
│   │   ├── tools_agent.py           # Agent 3 : Outils / MCP
│   │   ├── verification_agent.py    # Agent 4 : Vérification
│   │   ├── synthesis_agent.py       # Agent 5 : Synthèse finale
│   │   └── example_custom_agent.py  # Template pour nouveaux agents
│   │
│   ├── memory/
│   │   └── memory_manager.py        # Session + persistante (SQLite)
│   │
│   ├── mcp/
│   │   └── __init__.py              # Serveur MCP local
│   │
│   └── utils/
│       └── router.py                # Routeur dynamique
│
├── frontend/
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── src/
│       ├── main.jsx                 # Point d'entrée React
│       └── App.jsx                  # Interface complète
│
├── data/                            # Créé automatiquement
│   ├── chroma_db/                   # Base vectorielle RAG
│   └── memory.db                    # Historique SQLite
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Métriques et évaluation

### Métriques collectées par run

| Métrique | Agent | Description |
|---|---|---|
| `latency_ms` | Tous | Temps d'exécution par agent |
| `confidence_score` | Verification | Score global [0, 1] |
| `quality_score` | Verification | Qualité de la réponse [0, 1] |
| `consistency_check` | Verification | Cohérent / Partiel / Incohérent |
| `total_latency_ms` | Orchestrateur | Temps total de la requête |

### Endpoint statistiques

```
GET /api/stats
→ {
    "total_conversations": 42,
    "avg_confidence": 0.84,
    "avg_latency_ms": 3200,
    "total_sessions": 8
  }
```

### Questions de recherche supportées

1. **Architecture hiérarchique vs distribuée** : comparer les scores de confiance et latences
2. **Signaux de routage** : analyser `router_decision.reasoning` par type de question
3. **Stabilité** : mesurer `confidence_score` sur N requêtes du même type
4. **Taux d'échec** : compter `agent_results[*].success == false`

---

## Licence

MIT — Projet académique, usage libre avec attribution.
