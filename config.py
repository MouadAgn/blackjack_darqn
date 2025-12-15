# config.py
HYPERPARAMS = {
    "env_name": "ALE/Blackjack-v5",
    "image_size": 84,
    "batch_size": 32,
    "buffer_size": 10000,
    "lr": 0.0001,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 50000,
    "sequence_length": 8,  # IMPORTANT : Longueur de la séquence pour le LSTM
    "hidden_dim": 512,
    "target_update": 1000, # Fréquence de mise à jour du target network
    "total_timesteps": 500000
}