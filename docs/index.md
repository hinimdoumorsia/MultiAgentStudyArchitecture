# 👋 **Bienvenue sur la documentation de mon projet métier**

Bienvenue dans la documentation du projet :

**Orchestrateur multi-agents adaptatif pour l’assistance académique**

---

## **Informations générales**

- **Réalisé par :** Hinimdou Morsia Guitdam  
- **Encadré par :** Professeur Hajji Tarik  
- **École :** École Nationale Supérieure d’Arts et Métiers de Meknès (ENSAM)  
- **Filière :** Intelligence Artificielle et Technologies des Données  
- **Année :** 4ème année

---

## **Domaine du projet**

**Architecture agentique et systèmes multi-agents**

---

## **Description rapide**

Ce projet propose une architecture multi-agents capable de répartir intelligemment des tâches complexes entre plusieurs agents spécialisés, en comparant deux approches principales :

- architecture hiérarchique (chef d’orchestre)
- architecture distribuée (peer-to-peer)

avec un mécanisme de **routage dynamique basé sur le type de requête et la qualité attendue**.

---



# Table des matières

## 1. [Introduction](introduction.md)
- [1.1 Contexte général des systèmes multi-agents](introduction.md#11-contexte-général-des-systèmes-multi-agents)
- [1.2 Problématique de recherche](introduction.md#12-problématique-de-recherche)
- [1.3 Objectifs du projet](introduction.md#13-objectifs-du-projet)
- [1.4 Contributions attendues](introduction.md#14-contributions-attendues)

## 2. [État de l’art](etat_de_l_art.md)
- [2.1 Architectures multi-agents (hiérarchique, distribuée, graphe)](etat_de_l_art.md#21-architectures-multi-agents-hiérarchique-distribuée-graphe)
- [2.2 Orchestration des LLM](etat_de_l_art.md#22-orchestration-des-llm)
- [2.3 RAG et systèmes augmentés](etat_de_l_art.md#23-rag-et-systèmes-augmentés)
- [2.4 Routage intelligent dans les systèmes IA](etat_de_l_art.md#24-routage-intelligent-dans-les-systèmes-ia)

## 3. [Architecture globale du système](architecture_globale.md)
- [3.1 Vue d’ensemble du système proposé](architecture_globale.md#31-vue-densemble-du-système-proposé)
- [3.2 Diagramme d’architecture](architecture_globale.md#32-diagramme-darchitecture)
- [3.3 Composants principaux](architecture_globale.md#33-composants-principaux)
- [3.4 Flux de données entre agents](architecture_globale.md#34-flux-de-données-entre-agents)

## 4. [Agents du système](agents_du_systeme.md)
- [4.1 Agent de planification](agents_du_systeme.md#41-agent-de-planification)
- [4.2 Agent RAG (Retrieval-Augmented Generation)](agents_du_systeme.md#42-agent-rag-retrieval-augmented-generation)
- [4.3 Agent Tools (exécution d’outils)](agents_du_systeme.md#43-agent-tools-exécution-doutils)
- [4.4 Agent de vérification](agents_du_systeme.md#44-agent-de-vérification)
- [4.5 Agent de synthèse](agents_du_systeme.md#45-agent-de-synthèse)

## 5. [Architectures comparées](architectures_comparees.md)
- [5.1 Architecture hiérarchique (chef d’orchestre)](architectures_comparees.md#51-architecture-hiérarchique-chef-dorchestre)
- [5.2 Architecture distribuée / peer-to-peer](architectures_comparees.md#52-architecture-distribuée--peer-to-peer)
- [5.3 Comparaison des deux approches](architectures_comparees.md#53-comparaison-des-deux-approches)
- [5.4 Avantages et limites](architectures_comparees.md#54-avantages-et-limites)

## 6. [Mécanisme de routage intelligent](mecanisme_routage_intelligent.md)
- [6.1 Principe du routeur dynamique](mecanisme_routage_intelligent.md#61-principe-du-routeur-dynamique)
- [6.2 Signaux utilisés (incertitude, coût, charge, complexité)](mecanisme_routage_intelligent.md#62-signaux-utilisés-incertitude-coût-charge-complexité)
- [6.3 Stratégies de décision](mecanisme_routage_intelligent.md#63-stratégies-de-décision)
- [6.4 Visualisation du routage](mecanisme_routage_intelligent.md#64-visualisation-du-routage)

## 7. [Mémoire et infrastructure](memoire_et_infrastructure.md)
- [7.1 Mémoire de session](memoire_et_infrastructure.md#71-mémoire-de-session)
- [7.2 Mémoire persistante](memoire_et_infrastructure.md#72-mémoire-persistante)
- [7.3 Serveur MCP (Model Context Protocol)](memoire_et_infrastructure.md#73-serveur-mcp-model-context-protocol)
- [7.4 Interaction inter-agents (A2A)](memoire_et_infrastructure.md#74-interaction-inter-agents-a2a)

## 8. [Implémentation](implementation.md)
- [8.1 Stack technique](implementation.md#81-stack-technique)
- [8.2 Architecture du code](implementation.md#82-architecture-du-code)
- [8.3 Orchestrateur multi-agents](implementation.md#83-orchestrateur-multi-agents)
- [8.4 Gestion des erreurs et reprise](implementation.md#84-gestion-des-erreurs-et-reprise)

## 9. [Expérimentation](experimentation.md)
- [9.1 Protocole expérimental](experimentation.md#91-protocole-expérimental)
- [9.2 Types de requêtes testées](experimentation.md#92-types-de-requêtes-testées)
- [9.3 Mesures (latence, échec, qualité, stabilité)](experimentation.md#93-mesures-latence-échec-qualité-stabilité)
- [9.4 Résultats](experimentation.md#94-résultats)

## 10. [Analyse des résultats](analyse_des_resultats.md)
- [10.1 Comparaison des architectures](analyse_des_resultats.md#101-comparaison-des-architectures)
- [10.2 Interprétation des métriques](analyse_des_resultats.md#102-interprétation-des-métriques)
- [10.3 Discussion scientifique](analyse_des_resultats.md#103-discussion-scientifique)

## 11. [Conclusion et perspectives](conclusion_et_perspectives.md)
- [11.1 Synthèse des résultats](conclusion_et_perspectives.md#111-synthèse-des-résultats)
- [11.2 Réponse à la problématique](conclusion_et_perspectives.md#112-réponse-à-la-problématique)
- [11.3 Limites du système](conclusion_et_perspectives.md#113-limites-du-système)
- [11.4 Travaux futurs](conclusion_et_perspectives.md#114-travaux-futurs)

## 12. [Livrables](livrables.md)
- [Article scientifique](livrables.md#article-scientifique)
- [Code orchestrateur](livrables.md#code-orchestrateur)
- [Schémas d’architecture](livrables.md#schémas-darchitecture)
- [Interface de démonstration](livrables.md#interface-de-démonstration)