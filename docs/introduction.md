# **1. Introduction**

## **1.1 Contexte général des systèmes multi-agents**

Les systèmes multi-agents (SMA) occupent aujourd’hui une place importante dans le domaine de l’intelligence artificielle. Grâce à l’émergence des grands modèles de langage (LLMs), plusieurs frameworks tels que AutoGen, MetaGPT ou CAMEL permettent désormais à plusieurs agents intelligents de collaborer afin de résoudre des tâches complexes.

Dans ce contexte, différentes architectures de coordination ont été proposées pour organiser les interactions entre agents. Les deux principales architectures sont :

- l’architecture hiérarchique, basée sur un orchestrateur central chargé de coordonner les agents ;
- l’architecture distribuée (peer-to-peer), où les agents communiquent directement entre eux sans contrôle central.

Ces approches présentent chacune des avantages et des limites selon la nature des requêtes traitées.

---

## **1.2 Problématique de recherche**

Dans la majorité des travaux existants, le choix de l’architecture multi-agents est effectué de manière statique avant l’exécution du système. Cependant, les requêtes des utilisateurs peuvent être très variées : questions analytiques, mathématiques, comparatives, génération de code, etc.

Ainsi, une architecture performante pour une tâche donnée peut devenir moins efficace pour une autre. Cette problématique soulève une question importante :

> Comment sélectionner dynamiquement l’architecture multi-agents la plus adaptée en fonction de la requête utilisateur ?

Ce problème de routage dynamique reste encore peu exploré dans la littérature scientifique actuelle.

---

## **1.3 Objectifs du projet**

L’objectif principal de ce projet est de concevoir un système intelligent capable de sélectionner automatiquement l’architecture multi-agents la plus appropriée entre :

- une architecture hiérarchique ;
- une architecture distribuée.

Pour atteindre cet objectif, plusieurs sous-objectifs ont été définis :

- construire un dataset réel de questions académiques annotées ;
- extraire des caractéristiques linguistiques pertinentes ;
- entraîner plusieurs modèles de machine learning ;
- comparer leurs performances expérimentales ;
- intégrer le meilleur modèle dans une API de routage dynamique.

---

## **1.4 Contributions attendues**

Ce travail apporte plusieurs contributions principales :

- la formalisation d’un protocole d’évaluation permettant de déterminer l’architecture optimale selon une requête ;
- la création d’un pipeline de routage dynamique basé sur l’apprentissage supervisé ;
- la comparaison expérimentale de plusieurs classifieurs et baselines ;
- une étude d’ablation des représentations utilisées ;
- une analyse des erreurs et des limites du système proposé.

L’objectif final est de proposer une approche légère, explicable et efficace permettant d’améliorer la sélection d’architectures dans les systèmes multi-agents modernes.


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