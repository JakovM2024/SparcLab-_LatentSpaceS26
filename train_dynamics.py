import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from latent_encoder import Encoder, Dynamics

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20


def build_window_dataset(encoder: Encoder, data: dict, window: int):
    """Encode every episode and slide a window to build (z_window, action, z_target) tuples."""
    starts = data["episode_starts"]
    states = data["states"]
    actions = data["actions"]

    z_windows, action_list, z_targets = [], [], []

    for i in range(len(starts) - 1):
        ep_imgs = states[starts[i]:starts[i + 1]]   # (T, 84, 84, 3)
        ep_acts = actions[starts[i]:starts[i + 1]]  # (T, 1)

        if len(ep_imgs) <= window:
            continue

        # encode episode in mini-batches to avoid OOM
        with torch.no_grad():
            ep_latents = torch.cat([
                encoder(torch.from_numpy(ep_imgs[b:b + BATCH_SIZE]))
                for b in range(0, len(ep_imgs), BATCH_SIZE)
            ])  # (T, latent_size)

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

    safe_data = np.load("data/trajectories/safe_policy.npz")
    unsafe_data = np.load("data/trajectories/unsafe_policy.npz")

    print("Encoding episodes...")
    safe_windows, safe_acts, safe_targets = build_window_dataset(encoder, safe_data, dynamics.window)
    unsafe_windows, unsafe_acts, unsafe_targets = build_window_dataset(encoder, unsafe_data, dynamics.window)

    z_windows = torch.cat([safe_windows, unsafe_windows])
    act_all = torch.cat([safe_acts, unsafe_acts])
    z_targets = torch.cat([safe_targets, unsafe_targets])

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
