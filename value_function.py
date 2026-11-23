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
    """Safety critic Q(z_window, a): how safe it is to take a now and act safely after.

    n_heads = 1 (default, as in the LatentCBF paper): a single Q network.
    n_heads = 2: TD3 twin critics; the target uses the min of the two heads. That min biases Q
    downward, and with gamma near 1 the bias compounds through bootstrapping (run value_omega3
    drifted to V ~ -45 with a margin bounded by +-2), so it is off by default.
    """
    def __init__(self, latent_size: int = 16, window: int = WINDOW, hidden: int = 256,
                 a_max: float = A_MAX, n_heads: int = 1):
        super().__init__()
        assert n_heads in (1, 2)
        self.a_max = a_max
        self.n_heads = n_heads
        self.q1 = mlp(latent_size * window + 1, 1, hidden)
        if n_heads == 2:
            self.q2 = mlp(latent_size * window + 1, 1, hidden)

    def forward(self, z_window: torch.Tensor, action: torch.Tensor):
        """Returns a tuple with one Q estimate per head."""
        x = torch.cat([z_window.flatten(1), action / self.a_max], dim=-1)  # actions normalized to [-1, 1]
        if self.n_heads == 1:
            return (self.q1(x).squeeze(-1),)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_min(self, z_window: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Q itself for one head, the min over heads for two."""
        qs = self.forward(z_window, action)
        return qs[0] if len(qs) == 1 else torch.min(*qs)


def load_critic(path: str, latent_size: int = 16, device="cpu") -> Critic:
    """Load a critic checkpoint, inferring 1 vs 2 heads from its keys (older runs used 2)."""
    state = torch.load(path, weights_only=True, map_location=device)
    n_heads = 2 if any(k.startswith("q2.") for k in state) else 1
    critic = Critic(latent_size, n_heads=n_heads)
    critic.load_state_dict(state)
    return critic.to(device).eval()


def safety_value(actor: Actor, critic: Critic, z_window: torch.Tensor) -> torch.Tensor:
    """V(z) = Q(z, pi_safe(z)). V >= 0: safe under the safety policy; V < 0: doomed."""
    return critic.q_min(z_window, actor(z_window))
