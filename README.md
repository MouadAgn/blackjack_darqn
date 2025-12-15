# 🃏 Blackjack Atari Agent - DARQN + Attention

Ce projet implémente un agent d'Apprentissage par Renforcement Profond (Deep RL) capable de jouer au Blackjack sur l'environnement Atari (`ALE/Blackjack-v5`) de Gymnasium.

L'architecture utilisée est un **DARQN (Deep Attention Recurrent Q-Network)**. Elle est conçue pour traiter des informations visuelles partielles et séquentielles, ce qui est idéal pour le Blackjack où l'agent doit :
1. **Voir** les cartes (Vision via CNN).
2. **Se souvenir** des cartes précédentes (Mémoire via LSTM).
3. **Se focaliser** sur les zones importantes de l'écran (Mécanisme d'Attention).

## 📂 Structure du Projet

L'organisation des fichiers suit une architecture modulaire :

```text
blackjack_darqn/
│
├── checkpoints/             # Dossier de sauvegarde des modèles (.pth)
├── logs/                    # Logs pour TensorBoard (optionnel)
│
├── src/                     # Code source principal
│   ├── __init__.py
│   ├── model.py             # Architecture DARQN (CNN + Attention + LSTM)
│   ├── memory.py            # Replay Buffer Séquentiel (gère les séquences temporelles)
│   ├── agent.py             # Logique d'apprentissage (Loss, Backprop, Target Update)
│   └── utils.py             # Wrappers d'environnement (Preprocessing Atari)
│
├── config.py                # Hyperparamètres (Learning rate, Batch size, Gamma...)
├── main.py                  # Script pour lancer l'entraînement
├── test.py                  # Script pour tester et visualiser (GIF avec Heatmap)
├── requirements.txt         # Dépendances Python
└── README.md                # Documentation du projet

⚙️ Installation
1. Prérequis
Assurez-vous d'avoir Python 3.8+ installé.

2. Installation des dépendances
Installez les bibliothèques nécessaires, y compris les ROMs Atari :

Bash

pip install -r requirements.txt
Contenu du requirements.txt suggéré :

Plaintext

gymnasium[atari, accept-rom-license]
torch
torchvision
numpy
opencv-python
imageio
3. Préparation des dossiers
Créez les dossiers pour stocker les sauvegardes si ce n'est pas fait :

Bash

mkdir checkpoints
mkdir logs
🚀 Utilisation
1. Entraînement de l'Agent (main.py)
Pour lancer l'apprentissage depuis zéro. L'agent va explorer l'environnement, remplir sa mémoire et apprendre via le DARQN.

Bash

python main.py
Les modèles seront sauvegardés automatiquement dans checkpoints/ tous les X épisodes (ex: model_100.pth).

Note : L'entraînement sur pixels est long. Laissez tourner plusieurs heures pour obtenir des résultats probants.

2. Test et Visualisation (test.py)
Ce script charge un modèle entraîné, joue une partie et génère un GIF montrant ce que l'IA "regarde" grâce à la carte d'attention.

Ouvrez test.py.

Modifiez la ligne de chargement avec votre fichier .pth :

Python

# Exemple
run_test("checkpoints/model_1000.pth")
Lancez le script :

Bash

python test.py
Le résultat sera sauvegardé dans le fichier blackjack_attention.gif.