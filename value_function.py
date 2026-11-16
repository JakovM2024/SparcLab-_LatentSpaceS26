"""Stage 3 networks: safety critic Q(z_window, a) and safety actor pi_safe(z_window).

The RL state is the window of WINDOW consecutive latents (oldest first), the same input the
dynamics model uses. A single frame shows theta but not omega, and safety depends on both.
"""
import torch
import torch.nn as nn
from latent_encoder import WINDOW
from enviornment import Environment

A_MAX = float(Environment().action_space.high[0])  # actuator limit, single source of truth


def mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    """pi_safe(z_window) -> action in [-A_MAX, A_MAX], shape (B, 1)."""
    def __init__(self, latent_size: int = 16, window: int = WINDOW, hidden: int = 256, a_max: float = A_MAX):
        super().__init__()
        self.a_max = a_max
        self.net = mlp(latent_size * window, 1, hidden)

    def forward(self, z_window: torch.Tensor) -> torch.Tensor:
        return self.a_max * torch.tanh(self.net(z_window.flatten(1)))


class Critic(nn.Module):
    """Twin safety critics (TD3). Q(z_window, a): how safe it is to take a now and act safely after.

    Taking the min of the two heads in the target makes the estimate conservative, which is the
    right direction for safety (a lower value means less safe).
    """
    def __init__(self, latent_size: int = 16, window: int = WINDOW, hidden: int = 256, a_max: float = A_MAX):
        super().__init__()
        self.a_max = a_max
        self.q1 = mlp(latent_size * window + 1, 1, hidden)
        self.q2 = mlp(latent_size * window + 1, 1, hidden)

    def forward(self, z_window: torch.Tensor, action: torch.Tensor):
        x = torch.cat([z_window.flatten(1), action / self.a_max], dim=-1)  # actions normalized to [-1, 1]
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_min(self, z_window: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.forward(z_window, action))


def safety_value(actor: Actor, critic: Critic, z_window: torch.Tensor) -> torch.Tensor:
    """V(z) = Q(z, pi_safe(z)). V >= 0: safe under the safety policy; V < 0: doomed."""
    return critic.q_min(z_window, actor(z_window))
