# Régulation de trafic par agents Q-learning indépendants

Projet personnel — grille 2×2 d'intersections, chaque intersection est un
agent de renforcement autonome qui apprend son propre cycle de feux sans
communication centralisée.

## Le problème

Peut-on réduire la congestion dans un petit réseau routier en laissant
chaque intersection décider seule de son cycle de feux, sans coordination
explicite entre elles, uniquement à partir de ses propres files d'attente ?

## Approche

- **Environnement** (`environment.py`) : grille 2×2, 4 files d'attente par
  intersection (N/S/E/O), propagation du trafic vers les intersections
  voisines, arrivées aléatoires aux bords de la grille.
- **Agent** (`agent.py`) : Q-learning tabulaire indépendant (IQL), politique
  epsilon-greedy, état discrétisé en 4 niveaux de congestion par axe.
- **Entraînement** (`train.py`) : comparaison contre une baseline à cycle
  fixe (non adaptative), pour quantifier le gain réel de l'apprentissage.
- **Interface** (`app.py`) : tableau de bord Streamlit — vue opérationnelle
  de la grille, diagrammes de phase par intersection, diagnostic de
  convergence (couverture de l'espace d'état, évolution des valeurs Q).

## Résultat

Sur 800 épisodes d'entraînement, les agents Q-learning réduisent la
congestion moyenne par rapport à une régulation à cycle fixe classique
(le pourcentage exact dépend du seed et du nombre d'épisodes — relancez
`train.py` pour la valeur à jour, affichée en fin d'exécution).

## Démonstration

Voici le lien de la démonstration : https://projet-rl-tpznhf7b6qbkf2ntseqp8u.streamlit.app/

## Limites et pistes d'amélioration

- Les agents sont indépendants (IQL) : ils ne voient pas explicitement le
  comportement de leurs voisins, seulement l'effet indirect via leurs
  propres files d'attente. Une approche multi-agent coordonnée (ex. QMIX,
  observation partagée) pourrait mieux gérer les effets de bord entre
  intersections adjacentes.
- La grille est petite (2×2) et la dynamique du trafic simplifiée
  (pas de véhicules individuels, seulement des compteurs de file).
- Le taux d'apprentissage et le schéma d'epsilon-decay viennent d'essais
  manuels, pas d'une recherche d'hyperparamètres systématique.

## Structure

```
├── environment.py    # dynamique de la grille de trafic
├── agent.py          # agent Q-learning tabulaire
├── train.py           # entraînement + comparaison à la baseline
├── app.py            # tableau de bord Streamlit
└── requirements.txt
```
