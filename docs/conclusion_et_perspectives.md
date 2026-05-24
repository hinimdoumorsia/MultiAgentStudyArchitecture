# **11. Conclusion et perspectives**

## **11.1 Synthèse des résultats**
Dans ce travail, nous avons étudié le problème du routage dynamique dans les systèmes multi-agents, en formulant la sélection entre une architecture hiérarchique et une architecture distribuée comme une tâche de classification supervisée.

Les résultats expérimentaux montrent clairement que :

Le choix de l’architecture a un impact significatif sur la qualité des réponses générées, la consommation de ressources et le taux d’hallucination.
Aucune architecture n’est universellement supérieure : la performance dépend fortement de la nature de la requête.
Le modèle de régression logistique, malgré sa simplicité, obtient les meilleures performances globales avec une accuracy de 81,5% et un F1-macro de 0,75, surpassant les baselines simples et certaines approches plus complexes.
L’analyse d’ablation confirme que la combinaison TF-IDF + features linguistiques manuelles constitue le meilleur compromis entre performance et légèreté dans notre contexte expérimental.

Ces résultats valident l’hypothèse principale selon laquelle un routeur léger basé sur l’apprentissage supervisé peut efficacement approximer une politique de sélection d’architecture.

## **11.2 Réponse à la problématique**
La problématique initiale était la suivante :

comment sélectionner dynamiquement l’architecture multi-agent la plus adaptée à une requête utilisateur ?

Les résultats obtenus permettent de répondre de manière claire :

Le routage dynamique supervisé constitue une solution efficace pour adapter automatiquement l’architecture d’exécution en fonction du type de requête.

Il permet :

d’améliorer le compromis entre qualité de réponse et coût computationnel,
de réduire l’utilisation non optimale d’une architecture inadaptée,
et d’introduire une couche d’intelligence décisionnelle indépendante des modèles de langage utilisés.

Ainsi, le routage ne doit pas être vu comme un choix fixe d’architecture, mais comme une décision contextuelle apprenable.

## **11.3 Limites du système**
Malgré les résultats encourageants, plusieurs limites doivent être soulignées :

Taille limitée du dataset : avec seulement 107 exemples exploitables, les performances restent sensibles au bruit et aux biais d’annotation.
Déséquilibre des classes : la classe “distribuée” est sous-représentée, ce qui affecte la stabilité du modèle malgré l’utilisation de SMOTE.
Dépendance aux LLMs d’évaluation : les scores utilisés pour l’annotation reposent sur des modèles externes, introduisant une variabilité potentielle.
Overhead architectural : l’architecture distribuée basée sur EventBus introduit une complexité supplémentaire pouvant pénaliser les requêtes simples.
Généralisation limitée : les expériences ont été menées uniquement sur des questions académiques en français, ce qui limite la portée des conclusions à d’autres domaines ou langues.

## **11.4 Travaux futurs**

Plusieurs extensions naturelles de ce travail peuvent être envisagées :

Augmentation du dataset, en particulier pour la classe distribuée afin de réduire les déséquilibres et améliorer la robustesse du modèle.
Exploration de représentations sémantiques avancées (Sentence-BERT, embeddings contextuels) sur des datasets plus larges.
Extension du système vers un routage multi-architectures (au-delà de H et D), incluant des configurations hybrides.
Introduction d’un meta-routeur basé sur apprentissage par renforcement, capable d’optimiser directement la politique de sélection en fonction de récompenses liées au coût et à la qualité.
Intégration d’agents multimodaux (texte + vision + code), afin d’élargir le champ d’application du système.
Validation du cadre sur des domaines réels industriels ou scientifiques, pour tester la généralisation de l’approche.