# 10. Analyse des résultats

## 10.1 Comparaison des architectures

### Vue d'ensemble

Les deux architectures atteignent un niveau de performance global comparable — **~8,9/10 en qualité** — mais leurs profils de performance divergent significativement selon le type de tâche. C'est précisément cette divergence qui justifie empiriquement le routage dynamique.

### Efficacité computationnelle

| Métrique | Hiérarchique | Distribuée (P2P) | Avantage |
| :--- | :---: | :---: | :---: |
| Latence moyenne | ~3 200 ms | ~2 700 ms | **P2P −16 %** |
| Latence sur requête simple (*low*) | ~39 000 ms* | ~13 500 ms* | **P2P −65 %** |
| Consommation tokens (requête simple) | Plus élevée | Plus faible | **P2P** |
| Agents activés (routage sélectif) | 4/5 en moyenne | Pipeline complet | **H** |
| Coût sur tâches analytiques/math | Optimal | Surcoût de vérification | **H** |

> *Latences observées sur la requête de démonstration *"bonjour comment allez vous ?"* — la différence est amplifiée par le double passage verification+synthesis de l'architecture hiérarchique sur cette version du graphe.*

**Sur les requêtes simples (complexité *low*), l'architecture distribuée est jusqu'à ~2,9× plus rapide.** Ce gain s'explique par la réactivité événementielle du pipeline P2P qui ne force pas de passages séquentiels redondants.

**Sur les tâches analytiques et mathématiques**, l'architecture hiérarchique est plus économique en tokens grâce à son **routage sélectif** : le routeur interne désactive les agents inutiles (ex. `ToolsAgent` exclu si aucun calcul détecté), réduisant la consommation globale.

### Qualité et hallucination par type de tâche

| Type de tâche | Architecture optimale | Justification |
| :--- | :---: | :--- |
| Analytique | **Hiérarchique** | Flux séquentiel déterministe → hallucination quasi nulle |
| Mathématique | **Hiérarchique** | Décomposition symbolique structurée nécessaire |
| Code | **Distribuée** | Parallélisme naturel, moins de contraintes séquentielles |
| Comparative | **Hiérarchique** | Cohérence garantie par le contrôle centralisé |
| Générale / Factuelle | Variable | Dépend de la longueur et de la complexité |

### Stabilité et robustesse

L'architecture hiérarchique offre une **stabilité supérieure** sur les tâches complexes grâce à son point de contrôle unique. L'architecture distribuée présente une **instabilité plus marquée** sur les tâches analytiques et comparatives, liée à l'absence de coordinateur central et au risque de conditions de course sur l'`EventBus`.

---

## 10.2 Interprétation des métriques

### Corrélation complexité — architecture optimale

L'analyse du dataset de 107 questions révèle une corrélation nette entre la complexité de la requête et l'architecture qui lui convient le mieux :

| Complexité estimée | Proportion H optimale | Proportion D optimale | Interprétation |
| :---: | :---: | :---: | :--- |
| **low** | 55 % | 45 % | Avantage P2P sur la latence compense |
| **medium** | 80 % | 20 % | Structure hiérarchique plus adaptée |
| **high** | 88 % | 12 % | Décomposition séquentielle indispensable |

> Plus la complexité augmente, plus l'architecture hiérarchique s'impose — le contrôle centralisé devient critique pour maintenir la cohérence sur des raisonnements multi-étapes.

### Analyse du score de confiance

Le score de confiance moyen (`confidence_score` retourné par le `VerificationAgent`) est légèrement supérieur en architecture distribuée (0,87 vs 0,84). Ce résultat peut sembler contre-intuitif, mais s'explique par le fait que :

1. L'architecture distribuée tend à **traiter plus vite les requêtes simples** (où la confiance est naturellement élevée)
2. L'architecture hiérarchique traite proportionnellement **plus de requêtes complexes** (où la confiance est plus basse par construction)
3. Le **score de confiance brut n'est pas un indicateur de qualité absolue** — il reflète l'auto-évaluation du VerificationAgent, qui peut être calibrée différemment selon le chemin d'exécution

### Analyse du score de qualité (Q)

Les deux architectures atteignent ~8,9/10 en qualité moyenne — un niveau élevé dans les deux cas. La différence de performance se manifeste **non pas dans le score moyen global, mais dans la variance par catégorie** :

- **Hiérarchique** : variance faible sur analytique/math (robustesse), variance plus élevée sur code
- **Distribuée** : variance faible sur code (robustesse), variance élevée sur analytique/comparatif

C'est cette **variance différentielle par catégorie** qui constitue le signal d'entraînement du Meta-Router, et non une différence de moyenne globale.

### Interprétation du taux d'hallucination

| Catégorie | Hiérarchique | Distribuée | Explication |
| :--- | :---: | :---: | :--- |
| Analytique | **Quasi nul** | Modéré–élevé | Le flux séquentiel force la cohérence étape par étape |
| Mathématique | **Quasi nul** | Modéré | La vérification centralisée détecte les erreurs numériques |
| Code | Modéré | **Faible** | Le parallélisme P2P convient aux tâches non séquentielles |
| Comparative | **Faible** | Modéré–élevé | La synthèse centralisée harmonise les points de vue |

> L'hallucination en architecture distribuée sur les tâches analytiques s'explique par l'absence de coordinateur : si deux agents produisent des résultats contradictoires, aucun composant central ne les réconcilie avant la synthèse.

### Ce que les métriques ne disent pas

Deux biais doivent être gardés en tête lors de l'interprétation :

**Biais d'annotation** : les scores `Sglobal` dépendent des pondérations empiriques (`Wq=0,40`, etc.). Des pondérations différentes pourraient inverser l'annotation de certaines questions proches du seuil `ε=0,05`.

**Dépendance aux LLMs** : les scores de qualité et d'hallucination sont produits par des modèles Claude distants. Une mise à jour de ces modèles pourrait modifier les annotations et donc les performances du Meta-Router.

---

## 10.3 Discussion scientifique

### Validation de l'hypothèse centrale

L'hypothèse principale de ce projet est que **le routage hybride dynamique est optimal pour l'assistance académique** — c'est-à-dire qu'un système capable de sélectionner l'architecture selon la requête surpasse tout système figé sur une seule architecture.

Les résultats expérimentaux **valident cette hypothèse** à trois niveaux :

**1. Aucune architecture n'est universellement préférable.**
Les scores de qualité sont équivalents en moyenne (~8,9/10), mais les profils par catégorie divergent significativement. Choisir toujours H ou toujours D conduit à des sous-performances systématiques sur certains types de requêtes.

**2. Le classifieur supervisé apporte un gain mesuré et significatif.**
La Régression Logistique atteint F1-macro = 0,747 contre 0,426 pour la meilleure baseline (Always-H), soit **+0,321 de gain** — un saut qualitatif qui confirme que les patterns de routage sont apprenables depuis les features linguistiques de la requête.

**3. Le routage est léger, explicable et sans hallucination.**
Contrairement aux approches Mixture-of-Experts [7] ou LLM-routeur [8], le classifieur supervisé retenu (Logistic Regression) opère en **< 1 ms à l'inférence**, est totalement interprétable et ne produit aucune hallucination de routage.

### Positionnement par rapport à la littérature

| Approche | Latence routeur | Explicabilité | Hallucination | Coût GPU |
| :--- | :---: | :---: | :---: | :---: |
| **Notre Meta-Router** ✅ | < 1 ms | ✅ Totale | ✅ Aucune | ✅ Aucun |
| Mixture-of-Experts [7] | ~10–100 ms | ❌ Faible | Possible | ❌ Élevé |
| LLM-routeur [8] | ~500–2000 ms | ❌ Limitée | ✅ Possible | ❌ Modéré |
| Always-H (baseline) | 0 ms | ✅ Triviale | ✅ Aucune | ✅ Aucun |
| Rule-Based (baseline) | < 1 ms | ✅ Partielle | ✅ Aucune | ✅ Aucun |

> Notre approche offre le **meilleur compromis** entre performance (F1=0,747), légèreté (< 1 ms), explicabilité et absence de risque d'hallucination de routage.

### Limites identifiées et pistes d'amélioration

**Limite 1 — Déséquilibre de classes.**
La classe distribuée ne compte que 26 exemples (24,3 % du dataset), ce qui explique son F1 plus faible (0,62 vs 0,88 pour hiérarchique). Malgré SMOTE, ce déséquilibre reste la principale source d'erreur. La priorité est d'**enrichir le dataset** avec des questions de programmation variées.

**Limite 2 — Généralisation domaine/langue.**
Le dataset est exclusivement composé de questions académiques en français. La généralisation à d'autres domaines (médical, juridique, industriel) ou d'autres langues (anglais, arabe) nécessite une validation spécifique et une nouvelle phase de collecte.

**Limite 3 — Représentation sémantique.**
TF-IDF capture bien les patterns lexicaux mais reste une représentation de surface. Sentence-BERT pourrait surpasser TF-IDF à partir de ~500 exemples annotés — une direction à explorer lors du prochain cycle de collecte.

**Limite 4 — Deux architectures seulement.**
Le routeur binaire (H vs D) pourrait être étendu à un **routeur multi-classes** incluant un troisième paradigme hybride (ex. hiérarchique avec parallélisme partiel), offrant une granularité de sélection plus fine.

### Perspectives

Ce travail ouvre plusieurs directions de recherche :

- **Routage en ligne** (*online learning*) : mettre à jour le classifieur en continu à partir des nouvelles exécutions enregistrées dans `memory.db`, sans réentraînement complet
- **LLM-judge calibré** : remplacer l'annotation semi-manuelle par un juge LLM dont les biais sont explicitement mesurés et corrigés, pour accélérer la collecte de données
- **Routage multi-architectures** : étendre la classification à N > 2 architectures avec un softmax multi-classes
- **Transfert de domaine** : fine-tuner le Meta-Router sur des datasets spécialisés (médical, juridique) à partir du modèle académique actuel comme point de départ

### Conclusion de l'analyse

Les expériences menées sur 320 exécutions et 107 questions annotées confirment que le **routage dynamique supervisé est une approche viable et efficace** pour la sélection d'architecture dans les systèmes multi-agents académiques. La Régression Logistique, entraînée sur des features TF-IDF + manuelles, offre un gain de +0,321 en F1-macro par rapport à la meilleure stratégie statique, avec une inférence quasi-instantanée et une explicabilité totale.

Ces résultats constituent une **preuve de concept solide** pour un système de routage adaptatif dans les SMA à base de LLMs, tout en identifiant clairement les axes prioritaires d'amélioration : enrichissement du dataset pour la classe distribuée, et exploration des embeddings sémantiques profonds sur corpus plus large.



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

