# 9. Expérimentation

## 9.1 Protocole expérimental

L'évaluation repose sur un protocole formalisé conçu pour garantir la comparabilité stricte des deux architectures. Les deux architectures partagent les **mêmes cinq agents**, les **mêmes prompts** et le **même environnement d'exécution** — la seule variable différenciante est leur mode d'organisation.

### Collecte des données

| Paramètre | Valeur |
| :--- | :--- |
| Questions collectées | **160 par architecture** → 320 exécutions au total |
| Catégories couvertes | Analytique, Mathématique, Code, Comparative, Générale |
| Runs par question (variance) | 10 runs avec initialisations aléatoires différentes |
| Seuil d'ambiguïté (exclusion) | `ε = 0,05` → questions exclues si `\|Sglobal(H) − Sglobal(D)\| < ε` |
| **Dataset final retenu** | **107 questions valides** (81 → H, 26 → D) |

### Scripts de collecte

```bash
# Collecte simultanée sur les deux architectures
python data_collection/collector.py

# Collecte ciblée — architecture hiérarchique uniquement
python data_collection/collector_hier.py

# Collecte ciblée — architecture distribuée uniquement
python data_collection/collector_dist.py
```

### Division entraînement / test

```
107 questions
    ├── Entraînement : 75 % (stratifié) → 80 questions
    └── Test         : 25 % (stratifié) → 27 questions
```

La stratification garantit que les proportions de classes (75,7 % H / 24,3 % D) sont **conservées dans les deux ensembles**.

---

## 9.2 Types de requêtes testées

Les 160 questions par architecture couvrent cinq catégories représentatives de l'usage académique réel :

| Catégorie | Description | Architecture privilégiée |
| :--- | :--- | :--- |
| **Analytique** | Raisonnement logique, décomposition en sous-étapes, analyse conceptuelle | Hiérarchique |
| **Mathématique** | Calculs symboliques, résolution d'équations, formules | Hiérarchique |
| **Code** | Génération de fonctions, débogage, explication d'algorithmes Python | Distribuée |
| **Comparative** | Mise en relation de concepts, différences/similitudes, tableaux | Hiérarchique |
| **Générale** | Questions factuelles, définitions, questions de politesse | Variable |

### Exemples de requêtes par catégorie

```
[Analytique]    "Explique le principe de la récursivité et ses limites."
[Mathématique]  "Résous l'équation 5x² + 3x − 2 = 0."
[Code]          "Écris une fonction Python qui vérifie si un nombre est premier."
[Comparative]   "Quelle est la différence entre une fonction et une méthode ?"
[Générale]      "Qu'est-ce que le théorème de Bayes ?"
```

---

## 9.3 Mesures collectées

### Métriques par run

Pour chaque exécution `(qᵢ, A)`, quatre scores normalisés sont calculés et enregistrés dans `./data/memory.db` :

| Métrique | Source | Plage | Description |
| :--- | :--- | :---: | :--- |
| **Qualité Q** | VerificationAgent | 0–10 | Cohérence, précision factuelle, structure, complétude |
| **Hallucination H** | VerificationAgent | 0–1 | Pénalise erreurs factuelles et incohérences |
| **Tokens Stok** | Tous les agents | normalisé | Efficacité computationnelle |
| **Outils Stools** | ToolsAgent | normalisé | Pertinence et utilisation des outils |
| **Latence totale** | Runner | ms | Temps bout-en-bout de la requête |
| **Latence par agent** | Chaque agent | ms | Temps d'exécution individuel |
| **Confiance** | VerificationAgent | 0–1 | Score de fiabilité global |
| **Architecture** | Runner | — | `"hierarchical"` ou `"p2p"` |

### Score global pondéré

```
Sglobal = Wt·St + Wtok·Stok + Wh·Sh + Wtools·Stools + Wq·(Q/10)
```

| Wt | Wtok | Wh | Wtools | **Wq** |
| :---: | :---: | :---: | :---: | :---: |
| 0,15 | 0,15 | 0,15 | 0,15 | **0,40** |

> La qualité reçoit le poids le plus élevé (0,40) car elle capture le mieux la valeur perçue de la réponse.

### Accès aux métriques en temps réel

```bash
# Statistiques globales via l'API
GET http://localhost:8000/api/stats
```

```json
{
  "total_conversations": 42,
  "by_architecture": {
    "hierarchical": { "count": 21, "avg_confidence": 0.84, "avg_latency_ms": 3200 },
    "p2p":          { "count": 21, "avg_confidence": 0.87, "avg_latency_ms": 2700 }
  },
  "meta_router_accuracy": 0.76
}
```

```python
# Accès programmatique depuis Python
from backend.memory.memory_manager import memory_manager

stats = memory_manager.get_stats()
# → {"total_conversations": 42, "avg_confidence": 0.84, "avg_latency_ms": 3200.1, "total_sessions": 7}
```

---

## 9.4 Résultats

### Performances globales par architecture

| Métrique | Hiérarchique | Distribuée (P2P) | Δ |
| :--- | :---: | :---: | :---: |
| Score de qualité moyen | ~8,9 / 10 | ~8,9 / 10 | ≈ 0 |
| Confiance moyenne | 0,84 | 0,87 | +0,03 P2P |
| Latence moyenne | ~3 200 ms | ~2 700 ms | **−500 ms P2P** |
| Taux d'hallucination (analytique/math) | **Quasi nul** | Plus élevé | H meilleur |
| Performance sur tâches de code | Moins adaptée | **Mieux adaptée** | P2P meilleur |
| Stabilité sur tâches comparatives | **Stable** | Instabilité marquée | H meilleur |

> **Constat principal :** aucune architecture n'est universellement préférable. La différence de performance est liée au **type de tâche**, pas à une supériorité absolue.

### Taux d'hallucination par catégorie

| Catégorie | Hiérarchique | Distribuée |
| :--- | :---: | :---: |
| Analytique | **Quasi nul** | Élevé |
| Mathématique | **Quasi nul** | Modéré |
| Code | Modéré | **Faible** |
| Comparative | **Stable** | Instable |
| Générale | Faible | Faible |

> L'architecture hiérarchique réduit fortement l'hallucination sur les tâches analytiques et mathématiques grâce à son flux séquentiel déterministe. L'architecture distribuée est plus efficace sur les tâches de code grâce à son parallélisme naturel.

### Performances du Meta-Router (classification supervisée)

| Modèle | Acc. test | F1 macro | ROC-AUC | Gap |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** ✅ | **0,815** | **0,747** | 0,714 | −0,156 |
| CatBoost | 0,815 | 0,667 | 0,686 | +0,109 |
| Random Forest | 0,741 | 0,533 | 0,764 | +0,235 |
| XGBoost | 0,667 | 0,400 | 0,643 | +0,300 |
| Gradient Boosting | 0,704 | 0,451 | 0,671 | +0,312 |
| *Baseline Always-H* | *—* | *0,426* | *—* | *—* |
| *Baseline Rule-Based* | *—* | *0,512* | *—* | *—* |

**La Régression Logistique dépasse la meilleure baseline de +0,321 en F1-macro.** Le gap négatif (−0,156) confirme l'absence de surapprentissage.

### Rapport de classification par classe (Logistic Regression, n=27)

| Classe | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |
| Distribuée | 0,67 | 0,57 | **0,62** | 7 |
| Hiérarchique | 0,86 | 0,90 | **0,88** | 20 |
| **Accuracy** | | | **0,815** | 27 |
| Macro avg | 0,76 | 0,74 | **0,75** | 27 |

### Étude d'ablation — contribution des features

| Configuration Φ(q) | Acc. test | F1 macro |
| :--- | :---: | :---: |
| Features manuelles seules (8 dim.) | 0,704 | 0,581 |
| TF-IDF seul (150 dim.) | 0,778 | 0,693 |
| **TF-IDF + features manuelles (158 dim.)** ✅ | **0,815** | **0,747** |
| Sentence-BERT seul | 0,741 | 0,631 |
| Sentence-BERT + features manuelles | 0,778 | 0,712 |

> La combinaison TF-IDF + features manuelles est la **configuration optimale** dans les conditions étudiées. Sentence-BERT reste une piste sérieuse pour un dataset plus large (> 500 exemples).

### Analyse des 5 erreurs de classification

| # | Requête | Vrai | Prédit | Cause |
| :---: | :--- | :---: | :---: | :--- |
| 1 | "Résous 5x² + 3x − 2 = 0" | H | D | Formule détectée mais décomposition symbolique non identifiée |
| 2 | "Calcule log₁₀(1000)" | D | H | Question courte, mal distinguée par TF-IDF |
| 3 | "Comment fonctionne une boucle for en Python ?" | D | H | Question conceptuelle assimilée à analytique |
| 4 | "Écris une fonction Python pour vérifier un premier" | D | H | Code sous-représenté en classe distribuée |
| 5 | "Différence entre fonction et méthode ?" | H | D | Question hybride comparative/code |

**Cause principale :** faible effectif de la classe distribuée (26 exemples) malgré SMOTE. L'enrichissement de cette classe avec des questions de programmation variées est la priorité identifiée.

### Ressources de reproductibilité

| Ressource | Lien |
| :--- | :--- |
| Questions d'étude (160 brutes / architecture) | [📄 Google Drive](https://drive.google.com/file/d/1KxcRF8VK9NqW_yjPUW-WgKlcsN5eL6b4/view) |
| Dataset hiérarchique annoté | [📄 Google Drive](https://drive.google.com/file/d/1dcOwou6JVUA68kl5kPCj0jiz2jEOUPop/view) |
| Dataset distribué annoté | [📄 Google Drive](https://drive.google.com/file/d/1HHVlSkyogRWjRE2g1GrIuNCG4xcSZ1sb/view) |
| Notebook d'expérimentation complet | [📓 Google Drive](https://drive.google.com/file/d/1FDWvlUyVW47MFLkkxf3gtsI1Q7Rd7Zs3/view) |
| Meilleur modèle retenu (joblib) | [⚙️ Google Drive](https://drive.google.com/file/d/1WbaPRPV0YPI0Ex_daTexzFJF0g5arV27/view) |
| **Article scientifique complet** | [📄 Google Drive](https://drive.google.com/file/d/1g6c-XtSc2BTC71z2XeQZjtR5w92EpYwA/view?usp=sharing) |