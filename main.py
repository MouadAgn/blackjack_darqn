from src.utils import make_env
from src.agent import DARQNAgent
from src.memory import RecurrentReplayBuffer
from config import HYPERPARAMS
import torch

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement lancé sur : {device}")

    env = make_env(HYPERPARAMS["env_name"])
    
    # Initialisation de l'agent
    agent = DARQNAgent(
        input_shape=(1, 84, 84), 
        num_actions=env.action_space.n, 
        config=HYPERPARAMS, 
        device=device
    )
    
    # Initialisation de la mémoire (ATTENTION: j'ai ajouté 'device' ici qui manquait)
    memory = RecurrentReplayBuffer(
        HYPERPARAMS["buffer_size"], 
        HYPERPARAMS["sequence_length"],
        device
    )
    
    # Boucle d'épisodes
    for episode in range(1, HYPERPARAMS["total_timesteps"]): # ou 10000
        
        # Calcul de Epsilon (Décroissance linéaire)
        # On réduit la part d'aléatoire au fur et à mesure que l'agent apprend
        epsilon = max(
            HYPERPARAMS["epsilon_end"], 
            HYPERPARAMS["epsilon_start"] - (episode / HYPERPARAMS["epsilon_decay"])
        )

        obs, _ = env.reset()
        
        # Initialisation de l'état caché du LSTM à zéro pour le début de la partie
        hidden = agent.policy_net.init_hidden(batch_size=1, device=device)
        
        episode_record = []
        done = False
        total_reward = 0
        
        while not done:
            # 1. Préparer l'observation pour le réseau
            # obs est (1, 84, 84), on ajoute la dimension Batch -> (1, 1, 84, 84)
            state_tensor = obs.unsqueeze(0).to(device)
            
            # 2. Choisir une action
            action, next_hidden = agent.select_action(state_tensor, hidden, epsilon)
            
            # 3. Exécuter l'action dans l'environnement
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 4. Stocker la transition dans l'historique temporaire de l'épisode
            # (obs, action, reward, next_obs, done)
            episode_record.append((obs, action, reward, next_obs, done))
            
            # 5. Apprentissage (Une étape d'optimisation)
            # On n'apprend que si on a assez de données en mémoire
            loss = agent.train_step(memory)
            
            # 6. Mise à jour pour l'étape suivante
            obs = next_obs
            hidden = next_hidden # Important : on passe le hidden state au tour suivant
            total_reward += reward
            
        # Fin de l'épisode : on ajoute tout l'historique dans la mémoire principale
        memory.push(episode_record)
        
        # Mise à jour du réseau cible (Target Network) périodiquement
        if episode % HYPERPARAMS["target_update"] == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        # Logs et sauvegarde
       # Modifiez 100 en 10 ou 1 pour le test
        if episode % 10 == 0: 
            print(f"Episode {episode} | Reward: {total_reward} | Epsilon: {epsilon:.2f} | Loss: {loss if loss else 0:.4f}")
            # Gardez la sauvegarde à 100 pour ne pas remplir votre disque
            if episode % 100 == 0:
                 torch.save(agent.policy_net.state_dict(), f"checkpoints/model_{episode}.pth")

if __name__ == "__main__":
    train()