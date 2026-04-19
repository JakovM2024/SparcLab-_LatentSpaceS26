import numpy as np
import torch
import matplotlib.pyplot as plt
from latent_encoder import Encoder, Decoder

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
    safe = np.load("data/trajectories/safe_policy.npz")
    unsafe = np.load("data/trajectories/unsafe_policy.npz")
    all_states = np.concatenate([safe["states"], unsafe["states"]], axis=0)
    indices = np.random.choice(len(all_states), NUM_EXAMPLES, replace=False)
    return all_states[indices]  # (NUM_EXAMPLES, 84, 84, 3) uint8


def reconstruct(encoder, decoder, images):
    with torch.no_grad():
        batch = torch.from_numpy(images)
        z = encoder(batch)
        out = decoder(z).numpy()  # (N, 84, 84, 3) float [0, 1]

    print(f"Reconstructed pixel range: min={out.min():.3f}  max={out.max():.3f}  mean={out.mean():.3f}")

    # convert to uint8 for unambiguous display
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def show(images, reconstructed):
    fig, axes = plt.subplots(2, NUM_EXAMPLES, figsize=(NUM_EXAMPLES * 2, 5))
    fig.suptitle("Top: original    Bottom: reconstructed", fontsize=13)

    for i in range(NUM_EXAMPLES):
        axes[0, i].imshow(images[i])
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    encoder, decoder = load_models()
    images = load_random_images()
    reconstructed = reconstruct(encoder, decoder, images)
    show(images, reconstructed)
