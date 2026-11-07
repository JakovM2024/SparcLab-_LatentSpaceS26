import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from latent_encoder import Encoder, Dynamics
from data_collectionINVP import POLICY_TYPES

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20
DATASETS = POLICY_TYPES  # pd, noisy_pd, random, pd_then_random


def build_window_dataset(encoder: Encoder, data, window: int):
    """Encode every episode and slide a window to build (z_window, action, z_target) tuples.

    Episode i has frames obs[f0:f1] (T+1 of them) and actions actions[a0:a1] (T of them),
    where actions[a0 + t] moves frame t to frame t + 1. For a target frame t:
        z_window = z[t - window : t]      (the `window` frames before t)
        action   = actions[a0 + t - 1]    (the action that produced frame t)
        z_target = z[t]
    """
    frame_starts = data["episode_starts"]
    action_starts = data["action_starts"]
    obs = data["obs"]
    actions = data["actions"]

    z_windows, action_list, z_targets = [], [], []

    for i in range(len(frame_starts) - 1):
        ep_imgs = obs[frame_starts[i]:frame_starts[i + 1]]          # (T+1, 84, 84, 3)
        ep_acts = actions[action_starts[i]:action_starts[i + 1]]    # (T, 1)
        assert len(ep_imgs) == len(ep_acts) + 1

        if len(ep_imgs) <= window:
            continue

        # encode episode in mini-batches to avoid OOM
        with torch.no_grad():
            ep_latents = torch.cat([
                encoder(torch.from_numpy(ep_imgs[b:b + BATCH_SIZE]))
                for b in range(0, len(ep_imgs), BATCH_SIZE)
            ])  # (T+1, latent_size)

        for t in range(window, len(ep_imgs)):
            z_windows.append(ep_latents[t - window:t])                              # (window, latent)
            action_list.append(torch.tensor(ep_acts[t - 1], dtype=torch.float32))  # (1,)
            z_targets.append(ep_latents[t])                                          # (latent,)

    return (
        torch.stack(z_windows),    # (N, window, latent_size)
        torch.stack(action_list),  # (N, 1)
        torch.stack(z_targets),    # (N, latent_size)
    )


if __name__ == "__main__":
    encoder = Encoder(latent_size=16)
    encoder.load_state_dict(torch.load("data/models/encoder.pt", weights_only=True))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    dynamics = Dynamics(latent_size=16)

    print("Encoding episodes...")
    parts = [
        build_window_dataset(encoder, np.load(f"data/trajectories/{name}_policy.npz"), dynamics.window)
        for name in DATASETS
    ]
    z_windows = torch.cat([p[0] for p in parts])
    act_all = torch.cat([p[1] for p in parts])
    z_targets = torch.cat([p[2] for p in parts])

    dataset = TensorDataset(z_windows, act_all, z_targets)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(dynamics.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print(f"Training dynamics on {len(dataset)} windows...")
    dynamics.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for z_win, act, z_tgt in loader:
            z_pred = dynamics(z_win, act)
            loss = loss_fn(z_pred, z_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}  loss: {total_loss / len(loader):.6f}")

    torch.save(dynamics.state_dict(), "data/models/dynamics.pt")
    print("Dynamics saved.")
