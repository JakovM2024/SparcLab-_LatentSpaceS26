import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

safe_data = np.load("data/trajectories/safe_policy.npz")
unsafe_data = np.load("data/trajectories/unsafe_policy.npz")

starts = safe_data["episode_starts"]
ep1_states = safe_data["safe"][starts[0]:starts[1]]

class Encoder(nn.Module):
    def __init__(self, state_size = 4, latent_size = 8, hidden_size = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU,
            nn.Linear(hidden_size, latent_size)
        )
    def forward(self, state):
        return self.network(state)

