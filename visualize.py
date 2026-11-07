import numpy as np
import torch
import matplotlib.pyplot as plt
from latent_encoder import Encoder, Decoder
from data_collectionINVP import POLICY_TYPES

NUM_EXAMPLES = 8
LATENT_SIZE = 16


def load_models():
    encoder = Encoder(latent_size=LATENT_SIZE)
    decoder = Decoder(latent_size=LATENT_SIZE)
    encoder.load_state_dict(torch.load("data/models/encoder.pt", weights_only=True))
    decoder.load_state_dict(torch.load("data/models/decoder.pt", weights_only=True))
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def load_random_images():
    """Sample frames across all policy datasets; also return the true angle of each."""
    images, thetas = [], []
    for name in POLICY_TYPES:
        data = np.load(f"data/trajectories/{name}_policy.npz")
        images.append(data["obs"])
        thetas.append(data["theta"])
    images, thetas = np.concatenate(images), np.concatenate(thetas)
    indices = np.random.choice(len(images), NUM_EXAMPLES, replace=False)
    return images[indices], thetas[indices]  # (NUM_EXAMPLES, 84, 84, 3) uint8, (NUM_EXAMPLES,)


def reconstruct(encoder, decoder, images):
    with torch.no_grad():
        batch = torch.from_numpy(images)
        z = encoder(batch)
        out = decoder(z).numpy()  # (N, 84, 84, 3) float [0, 1]

    print(f"Reconstructed pixel range: min={out.min():.3f}  max={out.max():.3f}  mean={out.mean():.3f}")

    # convert to uint8 for unambiguous display
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def show(images, reconstructed, thetas):
    fig, axes = plt.subplots(2, NUM_EXAMPLES, figsize=(NUM_EXAMPLES * 2, 5))
    fig.suptitle("Top: original    Bottom: reconstructed", fontsize=13)

    for i in range(NUM_EXAMPLES):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f"θ = {thetas[i]:+.2f}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    encoder, decoder = load_models()
    images, thetas = load_random_images()
    reconstructed = reconstruct(encoder, decoder, images)
    show(images, reconstructed, thetas)
