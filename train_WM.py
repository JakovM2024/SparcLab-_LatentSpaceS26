import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import string
from latent_encoder import WorldModel

class Trainer():
    def __init__(self, safe_path: string, unsafe_path: string, epochs = 50):
        self.safe_data = np.load(safe_path)
        self.unsafe_data = np.load(unsafe_path)
        self.epochs = epochs
        
    def train(self, world_model: WorldModel):
        starts = self.safe_data["episode_starts"]
        num_episodes = starts.size
        optimizer = optim.Adam(list(world_model.encoder.parameters()) + list(world_model.decoder.parameters()) + list(world_model.dynamics.parameters()))


        for episode in range(num_episodes) - 1:
            states = self.safe_data[starts[episode]:starts[episode + 1]]
            for x in range(states.size):
                curr_state = torch.FloatTensor(states["states"][x])
                action = torch.FloatTensor(states["actions"][x])
                next_state = torch.FloatTensor(states["next_states"][x])
                state_reconstructed, z_predicted, z_actual = world_model(curr_state, next_state, action)
                recon_loss = nn.MSELoss()(curr_state,state_reconstructed)
                dynamics_loss = nn.MSELoss()(z_predicted,z_actual)
                loss = recon_loss + dynamics_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()






if __name__ == "__main__":
    trainee = Trainer("data/trajectories/safe_policy.npz", "data/trajectories/unsafe_policy.npz")
    starts = trainee.safe_data["episode_starts"]
    print(starts.size)