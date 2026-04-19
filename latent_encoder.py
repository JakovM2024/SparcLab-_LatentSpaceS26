import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Image input: (B, 84, 84, 3) uint8
# Encoder spatial sizes: 84 → 39 → 18 → 16, flatten = 64*16*16 = 16384

WINDOW = 3  # number of consecutive latents fed to dynamics

class Encoder(nn.Module):
    def __init__(self, latent_size: int = 16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=2),   # → (32, 39, 39)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64, 18, 18)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64, 16, 16)
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 16 * 16, latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).float() / 255.0  # (B, 3, 84, 84)
        x = self.cnn(x).flatten(1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_size: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 16 * 16),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),                    # → (64, 18, 18)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=1),  # → (32, 39, 39)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=8, stride=2),                     # → (3, 84, 84)
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 64, 16, 16)
        return self.deconv(x).permute(0, 2, 3, 1)  # → (B, 84, 84, 3)


class Dynamics(nn.Module):
    def __init__(self, latent_size: int = 16, action_size: int = 1, hidden_size: int = 128, window: int = WINDOW):
        super().__init__()
        self.window = window
        self.network = nn.Sequential(
            nn.Linear(latent_size * window + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

    def forward(self, z_window: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # z_window: (B, window, latent_size) → flatten to (B, window*latent_size)
        if action.dim() == 1:
            action = action.unsqueeze(-1)  # (B,) → (B, 1)
        x = torch.cat([z_window.flatten(1), action], dim=-1)
        return self.network(x)


class WorldModel(nn.Module):
    def __init__(self, latent_size: int = 16, action_size: int = 1, hidden_size: int = 128):
        super().__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)
        self.dynamics = Dynamics(latent_size, action_size, hidden_size)

    def predict(self, obs_window: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Given a window of images and an action, return the predicted next image."""
        B, W, H, HH, C = obs_window.shape
        z_window = self.encoder(obs_window.view(B * W, H, HH, C)).view(B, W, -1)
        z_next = self.dynamics(z_window, action)
        return self.decoder(z_next)
