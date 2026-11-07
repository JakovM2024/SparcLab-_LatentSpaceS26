import numpy as np
import torch
import matplotlib.pyplot as plt
from enviornment import Environment
from latent_encoder import Encoder, Margin

if __name__ == "__main__":
    encoder = Encoder(latent_size=16)
    encoder.load_state_dict(torch.load("data/models/encoder.pt", weights_only=True))
    margin = Margin(latent_size=16)
    margin.load_state_dict(torch.load("data/models/margin.pt", weights_only=True))
    encoder.eval()
    margin.eval()

    env = Environment()
    angles = np.linspace(-0.8, 0.8, 400)
    images = []
    for a in angles:
        env.state = np.array([a, 0], dtype=np.float32)
        images.append(env.render())

    with torch.no_grad():
        m = margin(encoder(torch.from_numpy(np.stack(images)))).numpy()

    plt.plot(angles, m, label="m(z)")
    plt.axhline(0, color="k", lw=0.8)
    for b in (-0.4, 0.4):
        plt.axvline(b, color="r", ls="--", lw=0.8)
    plt.xlabel("true angle θ (rad)")
    plt.ylabel("margin m(E(image))")
    plt.title("Learned margin vs. true failure boundary (red)")
    plt.legend()
    plt.show()