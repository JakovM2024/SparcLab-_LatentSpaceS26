import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#safe_data = np.load("data/trajectories/safe_policy.npz")
#unsafe_data = np.load("data/trajectories/unsafe_policy.npz")

#starts = safe_data["episode_starts"]
#ep1_states = safe_data["safe"][starts[0]:starts[1]]

class Encoder(nn.Module):
    def __init__(self, state_size: int = 2, latent_size: int = 8, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class Decoder(nn.Module):
    def __init__(self, state_size: int = 2, latent_size: int = 8, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class WorldModel(nn.Module):
    def __init__(self, state_size: int = 2, action_size: int =1,  latent_size: int = 8, hidden_size: int = 64):
        super().__init__()
        self.encoder = Encoder(state_size, latent_size, hidden_size)
        self.decoder = Decoder(state_size, latent_size, hidden_size)
        self.dynamics = Dynamics(latent_size, action_size, hidden_size)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_1 = self.dynamics(z)
        x_latent_reconstructed = self.decoder(z_1)
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    
class Dynamics(nn.Module):
    def __init__(self, latent_size: int = 8, action_size: int = 1, hidden_size: int=64 ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)

if __name__ == "__main__":
    rand_int = torch.rand(1,2)
    print(rand_int)
    model = WorldModel()
    rand_int = model.forward(rand_int)
    print(rand_int)
