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
    model.eval() # Mode évaluation
    return model

# Fonction pour superposer l'attention sur l'image du jeu
def visualize_attention(frame, attention_map):
    """
    frame: image RGB originale du jeu (210, 160, 3)
    attention_map: grille 7x7 sortie du modèle
    """
    # 1. Redimensionner l'attention (7x7) à la taille de l'écran (160x210)
    att_resized = cv2.resize(attention_map, (160, 210), interpolation=cv2.INTER_CUBIC)
    
    # 2. Normaliser entre 0 et 255
    att_resized = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8)
    att_uint8 = np.uint8(255 * att_resized)
    
    # 3. Appliquer une colormap (JET = Bleu vers Rouge)
    # OpenCV génère du BGR, il faut convertir en RGB pour aller avec Gymnasium
    heatmap = cv2.applyColorMap(att_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 4. Superposer : 60% image originale + 40% heatmap
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlay

def run_test(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # On crée l'environnement
    env = make_env(HYPERPARAMS["env_name"], render_mode='rgb_array')
    
    num_actions = env.action_space.n
    
    # Chargement sécurisé du modèle
    try:
        model = load_model(model_path, device, num_actions)
        print(f"Chargement du modèle depuis {model_path}...")
    except FileNotFoundError:
        print(f"ERREUR : Le fichier {model_path} n'existe pas encore.")
        print("Vérifiez que vous avez bien lancé main.py et attendu l'épisode 100.")
        return

    obs, _ = env.reset()
    hidden = model.init_hidden(batch_size=1, device=device)
    
    frames = [] 
    done = False
    
    print("Début de la partie...")
    
    # --- SÉCURITÉ : Compteur de pas pour éviter la boucle infinie ---
    steps = 0
    max_steps = 100 # On force l'arrêt si la partie dure trop longtemps
    
    while not done and steps < max_steps:
        steps += 1
        
        # --- 1. Préparation de l'observation ---
        # obs est déjà (1, 84, 84) grâce à utils.py
        # On ajoute la dimension Batch -> (1, 1, 84, 84)
        current_obs = obs.unsqueeze(0).to(device)
        
        # --- 2. Inférence ---
        with torch.no_grad():
            q_values, hidden, att_weights = model(current_obs, hidden)
            action = q_values.argmax().item()
            
        # --- 3. Visualisation ---
        raw_frame = env.render() # Image RGB
        
        # On récupère la carte d'attention
        att_map_numpy = att_weights.squeeze().cpu().numpy()
        
        # Création de l'image mixée
        mixed_frame = visualize_attention(raw_frame, att_map_numpy)
        frames.append(mixed_frame)
        
        # --- 4. Action dans le jeu ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            print(f"Partie terminée (Victoire/Défaite). Récompense finale : {reward}")
    
    if steps >= max_steps:
        print("Partie arrêtée (Limite de temps atteinte).")

    env.close()
    
    # Sauvegarde en GIF
    if len(frames) > 0:
        save_path = "blackjack_attention.gif"
        imageio.mimsave(save_path, frames, fps=15)
        print(f"Animation sauvegardée sous : {save_path}")
    else:
        print("Erreur : Aucune frame n'a été enregistrée.")

if __name__ == "__main__":
    # Assurez-vous que ce fichier existe bien dans le dossier checkpoints/
    run_test("checkpoints/model_100.pth")