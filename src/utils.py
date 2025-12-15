import gymnasium as gym
import ale_py
import shimmy
from gymnasium.wrappers import AtariPreprocessing
import torch
import numpy as np

# --- CLASSE PERSONNALISÉE POUR ÉVITER LE BUG DE VERSION ---
class ToTensor(gym.ObservationWrapper):
    """
    Convertit l'observation (Numpy Array) en PyTorch Tensor.
    Ajoute aussi la dimension du canal (Channel) : (84, 84) -> (1, 84, 84).
    """
    def __init__(self, env):
        super().__init__(env)
        # On ne définit pas observation_space ici pour éviter les conflits de types
        # entre Gym (qui veut du Numpy) et notre code (qui veut du Torch).

    def observation(self, observation):
        # 1. Conversion en Tensor Float
        tensor = torch.tensor(np.array(observation), dtype=torch.float32)
        
        # 2. Ajout de la dimension Channel si elle manque (H, W) -> (C, H, W)
        # AtariPreprocessing renvoie (84, 84), on veut (1, 84, 84)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
            
        return tensor

def make_env(env_name, render_mode=None):
    # 1. Création de l'environnement avec frameskip=1 pour éviter le conflit
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)
    
    # 2. Prétraitement Atari (Resize 84x84, Grayscale, Scaling)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=True)
    
    # 3. Conversion en Tensor via notre classe personnalisée (plus robuste que TransformObservation)
    env = ToTensor(env)
    
    return env