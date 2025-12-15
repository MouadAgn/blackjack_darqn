import torch
import gymnasium as gym
import numpy as np
import cv2  # OpenCV pour le traitement d'image
import imageio # Pour créer le GIF
from src.model import DARQN
from src.utils import make_env
from config import HYPERPARAMS

# Fonction pour charger le modèle
def load_model(path, device, num_actions):
    model = DARQN(input_shape=(1, 84, 84), num_actions=num_actions, hidden_dim=HYPERPARAMS["hidden_dim"])
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval() # Mode évaluation (désactive dropout, etc.)
    return model

# Fonction pour superposer l'attention sur l'image du jeu
def visualize_attention(frame, attention_map):
    """
    frame: image RGB originale du jeu (210, 160, 3)
    attention_map: grille 7x7 sortie du modèle
    """
    # 1. Redimensionner l'attention (7x7) à la taille de l'écran (160x210)
    # On utilise INTER_CUBIC pour avoir un effet "nuage" lisse plutôt que des carrés pixelisés
    att_resized = cv2.resize(attention_map, (160, 210), interpolation=cv2.INTER_CUBIC)
    
    # 2. Normaliser entre 0 et 255 pour la couleur
    att_resized = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8)
    att_uint8 = np.uint8(255 * att_resized)
    
    # 3. Appliquer une colormap (JET = Bleu vers Rouge)
    heatmap = cv2.applyColorMap(att_uint8, cv2.COLORMAP_JET)
    
    # 4. Superposer : 60% image originale + 40% heatmap
    # Convertir la frame RGB (Gym) en BGR (OpenCV) si besoin, ou inversement
    if frame.shape[2] == 3: # Si couleur
         # OpenCV utilise BGR par défaut, Matplotlib/Gym utilisent RGB.
         # Ici on suppose que frame est RGB.
         pass

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlay

def run_test(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # On crée l'environnement avec render_mode='rgb_array' pour récupérer les images couleurs
    env = make_env(HYPERPARAMS["env_name"], render_mode='rgb_array')
    
    num_actions = env.action_space.n
    model = load_model(model_path, device, num_actions)
    
    print(f"Chargement du modèle depuis {model_path}...")
    
    obs, _ = env.reset()
    hidden = model.init_hidden(batch_size=1, device=device)
    
    frames = [] # Pour stocker le GIF
    done = False
    
    print("Début de la partie...")
    
    while not done:
        # --- 1. Préparation de l'observation ---
        # Obs est un Tensor (1, 84, 84) grâce au wrapper
        # On ajoute la dimension Batch -> (1, 1, 84, 84)
        current_obs = obs.unsqueeze(0).to(device)
        
        # --- 2. Inférence ---
        with torch.no_grad():
            q_values, hidden, att_weights = model(current_obs, hidden)
            action = q_values.argmax().item()
            
        # --- 3. Visualisation ---
        # On récupère l'image brute couleur pour l'humain
        raw_frame = env.render() # (210, 160, 3)
        
        # On récupère la carte d'attention (1, 7, 7) -> on passe en numpy (7, 7)
        att_map_numpy = att_weights.squeeze().cpu().numpy()
        
        # Création de l'image mixée
        mixed_frame = visualize_attention(raw_frame, att_map_numpy)
        frames.append(mixed_frame)
        
        # --- 4. Action dans le jeu ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            print(f"Partie terminée. Récompense finale : {reward}")

    env.close()
    
    # Sauvegarde en GIF
    save_path = "blackjack_attention.gif"
    imageio.mimsave(save_path, frames, fps=15) # 15 images par seconde
    print(f"Animation sauvegardée sous : {save_path}")

if __name__ == "__main__":
    # Remplacez par le chemin de votre meilleur checkpoint
    # Si vous n'avez pas encore entraîné, le script plantera au chargement.
    # Pour tester le code sans entraînement, commentez 'model.load_state_dict...' dans load_model
    run_test("checkpoints/model_final.pth")