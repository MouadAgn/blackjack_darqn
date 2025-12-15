import torch
import torch.optim as optim
import random
import numpy as np
from .model import DARQN

class DARQNAgent:
    def __init__(self, input_shape, num_actions, config, device):
        self.device = device
        self.config = config
        self.num_actions = num_actions
        
        # 1. Initialisation des réseaux (Policy et Target)
        self.policy_net = DARQN(input_shape, num_actions, config["hidden_dim"]).to(device)
        self.target_net = DARQN(input_shape, num_actions, config["hidden_dim"]).to(device)
        
        # 2. Copie des poids du policy vers le target
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Le target net ne doit pas apprendre directement
        
        # 3. Optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["lr"])
        
    def select_action(self, state, hidden, epsilon):
        """
        Choisit une action selon la politique Epsilon-Greedy.
        state: Tensor (1, 1, 84, 84)
        hidden: Tuple (h, c)
        """
        if random.random() < epsilon:
            # Action aléatoire
            return random.randrange(self.num_actions), hidden
        else:
            # Action du réseau
            with torch.no_grad():
                q_values, next_hidden, _ = self.policy_net(state, hidden)
                return q_values.argmax().item(), next_hidden

    def train_step(self, memory):
        batch_size = self.config["batch_size"]
        if len(memory.buffer) < batch_size:
            return None # Pas assez de données
        
        # 1. Récupération des données (Batch, Seq, ...)
        # obs shape: (Batch, Seq, 1, 84, 84)
        obs, actions, rewards, next_obs, dones, masks = memory.sample(batch_size)
        
        # 2. Initialisation des états cachés
        # h_0, c_0 = zéros
        hidden_state = self.policy_net.init_hidden(batch_size, self.device)
        target_hidden_state = self.target_net.init_hidden(batch_size, self.device)
        
        loss = 0
        
        # 3. Boucle temporelle sur la séquence
        for t in range(self.config["sequence_length"]):
            # Extraction des données au pas de temps t
            # obs[:, t] -> prend toutes les lignes du batch, à l'instant t
            curr_obs = obs[:, t]       
            curr_action = actions[:, t].unsqueeze(1) 
            curr_reward = rewards[:, t]
            curr_next_obs = next_obs[:, t]
            curr_done = dones[:, t]
            curr_mask = masks[:, t] # Sert à ignorer le padding
            
            # --- Forward Policy Net ---
            # On passe l'obs actuelle et l'état caché précédent
            # q_values shape: (Batch, num_actions)
            q_values, hidden_state, _ = self.policy_net(curr_obs, hidden_state)
            
            # On récupère la Q-value de l'action qui a été réellement jouée
            q_val = q_values.gather(1, curr_action).squeeze(1)
            
            # --- Forward Target Net ---
            # Pour le "Next State", on n'a pas besoin de calculer les gradients (no_grad)
            with torch.no_grad():
                # On utilise target_hidden_state pour garder la cohérence temporelle du target
                target_q_values, target_hidden_state, _ = self.target_net(curr_next_obs, target_hidden_state)
                
                # Double DQN logic (optionnel mais meilleur) :
                # On pourrait utiliser policy_net pour choisir l'action max, mais restons simple (DQN classique) :
                next_q_val = target_q_values.max(1)[0]
                
                # Formule de Bellman
                target = curr_reward + (self.config["gamma"] * next_q_val * (1 - curr_done))
            
            # --- Calcul de la Loss pour ce pas de temps ---
            # On utilise le masque pour ne pas apprendre sur les zones de padding (zéros ajoutés)
            step_loss = (q_val - target.detach()).pow(2) * curr_mask
            
            # On accumule la loss moyenne sur le batch
            loss += step_loss.mean()
            
        # 4. Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradient pour éviter l'explosion (très important avec les LSTM/RNN)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()