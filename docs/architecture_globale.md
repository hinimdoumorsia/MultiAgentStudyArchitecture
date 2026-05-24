# 3. Architecture globale du système

## 3.1 Vue d'ensemble du système proposé

Dans le cadre de ce projet de recherche, nous avons conçu et implémenté deux architectures multi-agents complètes afin d'étudier leurs performances dans un contexte d'assistance académique intelligente.

L'objectif est de réaliser une étude expérimentale comparative rigoureuse entre deux approches :

- une **architecture hiérarchique**, pilotée par un orchestrateur central (`AcademicOrchestrator`),
- une **architecture distribuée**, basée sur la collaboration entre agents sans point de contrôle unique (`PeerToPeerRunner`).

Les deux systèmes utilisent les mêmes modèles de langage (LLMs locaux et via API), les mêmes invites, ainsi que les mêmes agents afin de garantir une comparaison équitable et scientifique. L'exposition de ces deux architectures se fait via une API REST construite avec **FastAPI**, servie sur le port `8000`. Un troisième mode, dit **dynamique**, a été introduit : un routeur à base de régression logistique (`DynamicRouter`) analyse la requête entrante et choisit automatiquement l'architecture la plus adaptée.

```python
# backend/main.py — Initialisation des deux architectures et du routeur
from backend.orchestrator import AcademicOrchestrator
from backend.distributed.peer_to_peer_runner import PeerToPeerRunner

orchestrator       = AcademicOrchestrator()   # Architecture HIÉRARCHIQUE
distributed_runner = PeerToPeerRunner()        # Architecture DISTRIBUÉE
dynamic_router     = DynamicRouter("models/final_router_pipeline.pkl")  # ROUTEUR

logger.info("✅ Architectures chargées : Hiérarchique + Distribuée")
```

---

## 3.2 Diagramme d'architecture

Le flux commence par un **Meta-Router** (`DynamicRouter`) qui analyse la requête utilisateur et la dirige vers l'une des deux architectures. Chaque architecture mobilise ensuite les mêmes agents spécialisés : Planning, RAG, Tools, Verification et Synthesis.

```
Requête utilisateur
        │
        ▼
┌──────────────────┐
│   DynamicRouter  │  ← Logistic Regression + TF-IDF
│  (Meta-Router)   │
└────────┬─────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐  ┌───────────┐
│ Orches-│  │ Peer-to-  │
│ trateur│  │ Peer      │
│ (Hiér.)│  │ (Distrib.)│
└────┬───┘  └─────┬─────┘
     │             │
     └──────┬──────┘
            ▼
  [Planning | RAG | Tools | Verification | Synthesis]
            │
            ▼
       Réponse finale
```

Les trois endpoints REST correspondants sont exposés comme suit :

```python
# Endpoint HIÉRARCHIQUE
@app.post("/api/query")
def query(request: QueryRequest):
    result = orchestrator.run(request.query, session_id)
    return result

# Endpoint DISTRIBUÉ
@app.post("/api/query/distributed")
def query_distributed(request: QueryRequest):
    result = distributed_runner.run(request.query, session_id)
    return result

# Endpoint DYNAMIQUE (routeur)
@app.post("/api/query/dynamic")
def query_dynamic(request: UnifiedQueryRequest):
    if not dynamic_router.is_available():
        raise HTTPException(503, "Routeur dynamique non disponible")
    result = dynamic_router.route(request.query, session_id)
    return result
```

---

## 3.3 Composants principaux

**1. Meta-Router (`DynamicRouter`)**

Le routeur est un pipeline `scikit-learn` sérialisé (`joblib`) combinant un vectoriseur TF-IDF et un modèle de régression logistique. Il extrait des features textuelles et numériques pour classifier la requête.

```python
class DynamicRouter:
    def __init__(self, model_path: str = "final_router_pipeline.pkl"):
        self.pipeline = joblib.load(self.model_path)
        self.confidence_threshold = 0.55

    def _extract_features(self, question: str) -> dict:
        features = {}
        features['len_chars']    = len(question)
        features['len_words']    = len(question.split())
        features['avg_word_len'] = features['len_chars'] / max(features['len_words'], 1)

        math_keywords = ['sin', 'cos', 'equation', 'derive', 'integrale', 'suite']
        features['math_keyword_count'] = sum(
            1 for kw in math_keywords if kw in question.lower()
        )
        features['has_formula'] = int(any(c in question for c in ['²', '^', '√', '=']))
        features['has_numbers'] = int(any(c.isdigit() for c in question))
        return features

    def predict(self, question: str):
        question_clean = self._clean_text(question)
        tfidf_vec    = self.pipeline['tfidf_vectorizer'].transform([question_clean]).toarray()
        feats        = self._extract_features(question)
        extra_vec    = np.array([[feats[col] for col in self.pipeline['feature_columns']]])
        extra_scaled = self.pipeline['feature_scaler'].transform(extra_vec)
        X_new = np.hstack([tfidf_vec, extra_scaled])

        pred_enc   = self.pipeline['model'].predict(X_new)[0]
        pred_proba = self.pipeline['model'].predict_proba(X_new)[0]
        architecture = self.pipeline['label_encoder'].inverse_transform([pred_enc])[0]
        confidence   = max(pred_proba)
        return architecture, confidence, {'hierarchique': pred_proba[1], 'distribue': pred_proba[0]}
```

**2. Orchestrateur LangGraph (Architecture Hiérarchique)**

`AcademicOrchestrator` est le point de contrôle central de l'architecture hiérarchique. Il reçoit la requête, coordonne séquentiellement les agents et consolide leur sortie en une réponse finale.

```python
# Appel depuis l'endpoint /api/query
result = orchestrator.run(request.query, session_id)
# → result contient : réponse, agent utilisé, durée, tokens, historique
```

**3. EventBus (Architecture Distribuée)**

`PeerToPeerRunner` implémente une architecture sans orchestrateur central. Les agents s'enregistrent sur un bus d'événements, émettent et consomment des messages de manière asynchrone, et s'auto-organisent pour produire la réponse.

```python
# Appel depuis l'endpoint /api/query/distributed
result = distributed_runner.run(request.query, session_id)
# → même interface de sortie que l'orchestrateur, pour comparabilité
```

**4. Agents Spécialisés**

Les cinq agents communs aux deux architectures sont : `PlanningAgent`, `RAGAgent`, `ToolsAgent`, `VerificationAgent` et `SynthesisAgent`. Ils sont accessibles via le registre central :

```python
# Récupération d'un agent depuis le registre (ex. RAG)
from backend.agents.registry import registry
rag = registry.get("rag")
rag.add_document(content, source, doc_id)
```

**5. Mémoire (SQLite / ChromaDB)**

Le module `memory_manager` gère deux niveaux de mémoire : une mémoire persistante (SQLite) pour l'historique de sessions, et une mémoire vectorielle (ChromaDB) pour la recherche sémantique.

```python
# Récupération de statistiques mémoire
@app.get("/api/stats")
def stats():
    return memory_manager.get_stats()

# Récupération de l'historique d'une session
@app.get("/api/history/{session_id}")
def history(session_id: str, limit: int = 10):
    return {
        "session_id": session_id,
        "history": memory_manager.persistent.get_session_history(session_id, limit),
    }
```

---

## 3.4 Flux de données entre agents

Le flux de données diffère selon l'architecture choisie, mais l'interface d'entrée/sortie reste identique pour permettre une comparaison objective.

**Architecture Hiérarchique** — état partagé `AcademicState` :

```
Requête → Orchestrateur → AcademicState (partagé)
              │
       ┌──────┴──────┐
       ▼             ▼
  PlanningAgent   RAGAgent
       │             │
       └──────┬──────┘
              ▼
         ToolsAgent
              │
              ▼
     VerificationAgent
              │
              ▼
       SynthesisAgent → Réponse finale
```

**Architecture Distribuée** — événements asynchrones via EventBus :

```
Requête → EventBus.emit("query")
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
PlanningAgent RAGAgent ToolsAgent
    │         │          │
    └─────────┼──────────┘
              ▼ EventBus.emit("partial_result")
     VerificationAgent
              │
              ▼ EventBus.emit("verified")
       SynthesisAgent → Réponse finale
```

Le routeur dynamique enrichit la réponse avec des métadonnées de routage, permettant une traçabilité complète de la décision :

```python
def route(self, question: str, session_id: str = None):
    architecture, confidence, probabilities = self.predict(question)
    logger.info(f"🔀 ROUTING: {question[:60]}... -> {architecture.upper()} (conf={confidence:.3f})")

    if architecture == "hierarchique":
        result = orchestrator.run(question, session_id)
    else:
        result = distributed_runner.run(question, session_id)

    # Ajout des métadonnées de routage à la réponse
    result['routing'] = {
        'architecture':  architecture,
        'confidence':    confidence,
        'probabilities': probabilities,
        'router_model':  self.pipeline['model_name']
    }
    return result
```

Ainsi, chaque réponse produite via `/api/query/dynamic` inclut non seulement la réponse académique, mais aussi la justification chiffrée du choix d'architecture (`confidence`, `probabilities`), ce qui facilite l'analyse comparative expérimentale.


##  **Accès au projet et exécution**

Si vous souhaitez exécuter le projet, poursuivre son exploration ou mieux comprendre l’ensemble de la documentation, les ressources complètes sont mises à disposition ci-dessous.

Ces éléments permettent de reproduire intégralement les expériences, d’exécuter les différentes architectures (hiérarchique, distribuée et dynamique) et de consulter les datasets ainsi que les résultats expérimentaux.

👉 Toutes les ressources nécessaires sont accessibles ici :


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