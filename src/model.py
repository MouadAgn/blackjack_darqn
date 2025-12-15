import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DARQN(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim=512):
        super(DARQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # 1. BLOC CNN (Feature Extractor)
        # On suppose une entrée standard Atari redimensionnée (C, H, W) = (1, 84, 84)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calcul de la taille de sortie du CNN pour l'attention
        # Pour une image 84x84, la sortie conv3 est 64 channels x 7 x 7
        self.conv_out_size = 7 * 7 
        self.feature_dim = 64 # Nombre de channels en sortie
        
        # 2. BLOC ATTENTION
        # Attention linéaire : combine les features de l'image et l'état caché du LSTM
        self.att_linear_features = nn.Linear(self.feature_dim, hidden_dim) 
        self.att_linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.att_linear_out = nn.Linear(hidden_dim, 1)
        
        # 3. BLOC LSTM
        # L'entrée du LSTM est le vecteur de contexte (features pondérées par l'attention)
        self.lstm = nn.LSTMCell(self.feature_dim, hidden_dim)
        
        # 4. BLOC DE SORTIE (Q-Values)
        self.fc_adv = nn.Linear(hidden_dim, num_actions) # Advantage
        self.fc_val = nn.Linear(hidden_dim, 1)           # Value (Dueling DQN structure is better)

    def forward(self, x, hidden_state):
        """
        x: Batch d'images (Batch, Channel, Height, Width)
        hidden_state: Tuple (h_t, c_t) du pas de temps précédent
        """
        batch_size = x.size(0)
        h_t, c_t = hidden_state
        
        # --- A. Extraction des Features ---
        features = F.relu(self.conv1(x))
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features)) 
        # features shape: (Batch, 64, 7, 7)
        
        # Aplatir spatialement pour l'attention : (Batch, 49, 64)
        # 49 zones spatiales (7x7), chaque zone a un vecteur de 64 caractéristiques
        features_flat = features.view(batch_size, -1, self.feature_dim) 
        
        # --- B. Calcul de l'Attention ---
        # On veut savoir quelle zone (parmi les 49) est importante sachant l'état caché h_t
        
        # Projection de l'état caché précédent (Batch, Hidden) -> (Batch, 1, Hidden)
        h_proj = self.att_linear_hidden(h_t).unsqueeze(1)
        
        # Projection des features (Batch, 49, 64) -> (Batch, 49, Hidden)
        f_proj = self.att_linear_features(features_flat)
        
        # Score d'attention (mécanisme additif classique style Bahdanau)
        att_score = torch.tanh(f_proj + h_proj) # Broadcasting de h_proj sur les 49 zones
        att_weights = F.softmax(self.att_linear_out(att_score), dim=1) # (Batch, 49, 1)
        
        # Création du vecteur de contexte (somme pondérée des features)
        # (Batch, 49, 64) * (Batch, 49, 1) -> Somme sur la dimension 1 -> (Batch, 64)
        context = torch.sum(features_flat * att_weights, dim=1)
        
        # --- C. Récurrence (LSTM) ---
        h_new, c_new = self.lstm(context, (h_t, c_t))
        
        # --- D. Calcul des Q-Values (Dueling) ---
        adv = self.fc_adv(h_new)
        val = self.fc_val(h_new)
        
        q_values = val + adv - adv.mean(1, keepdim=True)
        
        # On retourne aussi les poids d'attention pour la visualisation (optionnel mais utile)
        return q_values, (h_new, c_new), att_weights.view(batch_size, 7, 7)

    def init_hidden(self, batch_size, device):
        """Initialise l'état caché à zéro pour le début d'un épisode"""
        return (torch.zeros(batch_size, self.hidden_dim).to(device),
                torch.zeros(batch_size, self.hidden_dim).to(device))