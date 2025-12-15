import numpy as np
import random
import torch
from collections import deque

class RecurrentReplayBuffer:
    def __init__(self, capacity, sequence_length, device):
        self.buffer = deque(maxlen=capacity)
        self.seq_len = sequence_length
        self.device = device

    def push(self, episode):
        """
        Stocke un épisode complet.
        episode format: liste de tuples (obs, action, reward, next_obs, done)
        """
        self.buffer.append(episode)

    def sample(self, batch_size):
        # Initialisation des listes pour le batch
        b_obs, b_actions, b_rewards, b_next_obs, b_dones, b_masks = [], [], [], [], [], []

        for _ in range(batch_size):
            episode = random.choice(self.buffer)
            ep_len = len(episode)

            # --- 1. Découpage de la séquence ---
            # Si l'épisode est plus court que la séquence demandée (ex: Blackjack fini en 2 coups)
            if ep_len < self.seq_len:
                start_idx = 0
                actual_len = ep_len
            else:
                # Sinon on prend un morceau au hasard
                start_idx = np.random.randint(0, ep_len - self.seq_len + 1)
                actual_len = self.seq_len

            # Extraction
            seq = episode[start_idx : start_idx + self.seq_len]
            
            # Dézippage de la séquence
            # (Zip(*seq) transforme [(s,a,r...), (s,a,r...)] en ([s,s], [a,a], ...))
            s, a, r, ns, d = zip(*seq)

            # --- 2. Padding (Remplissage) ---
            # On doit avoir une taille fixe pour que PyTorch accepte le tenseur
            # On remplit avec des 0 si la séquence est trop courte
            pad_len = self.seq_len - actual_len
            
            # Création des tenseurs avec padding
            # obs shape: (Seq, 1, 84, 84)
            s_tensor = torch.stack(s)
            ns_tensor = torch.stack(ns)
            
            if pad_len > 0:
                # Zéros pour le padding
                zero_obs = torch.zeros(pad_len, 1, 84, 84)
                s_tensor = torch.cat([s_tensor, zero_obs])
                ns_tensor = torch.cat([ns_tensor, zero_obs])
                
                # Padding pour scalaires
                a = list(a) + [0] * pad_len
                r = list(r) + [0.0] * pad_len
                d = list(d) + [True] * pad_len # True pour ne pas apprendre sur le padding
                
                # Masque : 1.0 si vraie donnée, 0.0 si padding
                mask = [1.0] * actual_len + [0.0] * pad_len
            else:
                mask = [1.0] * self.seq_len

            b_obs.append(s_tensor)
            b_actions.append(torch.tensor(a, dtype=torch.long))
            b_rewards.append(torch.tensor(r, dtype=torch.float))
            b_next_obs.append(ns_tensor)
            b_dones.append(torch.tensor(d, dtype=torch.float))
            b_masks.append(torch.tensor(mask, dtype=torch.float))

        # Empilement final : (Batch, Sequence, ...)
        return (torch.stack(b_obs).to(self.device),
                torch.stack(b_actions).to(self.device),
                torch.stack(b_rewards).to(self.device),
                torch.stack(b_next_obs).to(self.device),
                torch.stack(b_dones).to(self.device),
                torch.stack(b_masks).to(self.device))