import numpy as np
import torch
import torch.optim as optim
from latent_encoder import Encoder, Margin


BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 50
GP_WEIGHT = 1.0  # λ in eq. (1): higher = smoother but more conservative boundary
DATASETS = ["safe", "unsafe", "demo"]

def encode(encoder, images):
    with torch.no_grad():
        return torch.cat([
            encoder(torch.from_numpy(images[b:b +BATCH_SIZE]))
            for b in range(0, len(images), BATCH_SIZE)
        ])

def load_labeled_latents(encoder):
    """safe[t] is decided by the angle AFTER step t, so it labels next_states[t]."""
    z_list, label_list = [], []
    for name in DATASETS:
        data = np.load(f"data/trajectories/{name}_policy.npz")
        z_list.append(encode(encoder, data["next_states"]))
        label_list.append(torch.from_numpy(data["safe"]))
    return torch.cat(z_list), torch.cat(label_list)

def gradient_penalty(margin, z_safe, z_unsafe):
    # z_hat: random points between safe/unsafe latent pairs, where the boundary lives
    alpha = torch.rand(len(z_safe), 1)
    z_hat = (alpha * z_safe + (1 - alpha) * z_unsafe).requires_grad_(True)
    grad = torch.autograd.grad(margin(z_hat).sum(), z_hat, create_graph=True)[0]
    return ((grad.norm(dim =1) - 1) ** 2 ).mean()

def margin_loss(margin, z_safe, z_unsafe):
    """Eq. (1): hinge on each class + gradient penalty for a bounded Lipschitz constant."""
    safe_term = torch.relu(1 - margin(z_safe)).mean()
    unsafe_term = torch.relu(1 + margin(z_unsafe)).mean()
    return safe_term + unsafe_term + GP_WEIGHT * gradient_penalty(margin, z_safe, z_unsafe)

if __name__ == "__main__":
    encoder = Encoder(latent_size=16)
    encoder.load_state_dict(torch.load("data/models/encoder.pt", weights_only = True))
    encoder.eval()

    print("Encoding frames...")
    z, safe = load_labeled_latents(encoder)
    z_safe, z_unsafe = z[safe], z[~safe]
    print(f"{len(z_safe)} safe / {len(z_unsafe)} unsafe latents")

    margin = Margin(latent_size=16)
    optimizer = optim.Adam(margin.parameters(), lr=LEARNING_RATE)
    steps_per_epoch = len(z_safe) // BATCH_SIZE

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            # sample each class separately: unsafe frames are only ~4% of the data
            s = z_safe[torch.randint(len(z_safe), (BATCH_SIZE,))]
            u = z_unsafe[torch.randint(len(z_unsafe), (BATCH_SIZE,))]
            loss = margin_loss(margin, s, u)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        with torch.no_grad():
            acc_safe = (margin(z_safe) > 0).float().mean().item()
            acc_unsafe = (margin(z_unsafe) < 0).float().mean().item()
        print(f"Epoch {epoch + 1}/{EPOCHS}  loss: {total_loss / steps_per_epoch:.4f}  "
              f"safe acc: {acc_safe:.3f}  unsafe acc: {acc_unsafe:.3f}")

    torch.save(margin.state_dict(), "data/models/margin.pt")
    print("Margin saved.")
    