import numpy as np
import torch
import matplotlib.pyplot as plt
from enviornment import Environment, FAIL_ANGLE, MAX_RENDER_ANGLE
from latent_encoder import Encoder, Margin
from data_collectionINVP import POLICY_TYPES

N_DATASET_POINTS = 3000  # dataset frames to scatter on top of the sweep
SEED = 0

if __name__ == "__main__":
    encoder = Encoder(latent_size=16)
    encoder.load_state_dict(torch.load("data/models/encoder.pt", weights_only=True))
    margin = Margin(latent_size=16)
    margin.load_state_dict(torch.load("data/models/margin.pt", weights_only=True))
    encoder.eval()
    margin.eval()

    # 1) sweep: render the pole at rest over the full renderable range
    env = Environment()
    angles = np.linspace(-MAX_RENDER_ANGLE, MAX_RENDER_ANGLE, 400)
    images = []
    for a in angles:
        env.state = np.array([a, 0], dtype=np.float32)
        images.append(env.render())

    with torch.no_grad():
        m_sweep = margin(encoder(torch.from_numpy(np.stack(images)))).numpy()

    # 2) real dataset frames, using the stored true angle and label
    rng = np.random.default_rng(SEED)
    obs, theta, safe = [], [], []
    for name in POLICY_TYPES:
        data = np.load(f"data/trajectories/{name}_policy.npz")
        obs.append(data["obs"]); theta.append(data["theta"]); safe.append(data["safe"])
    obs, theta, safe = np.concatenate(obs), np.concatenate(theta), np.concatenate(safe)
    idx = rng.choice(len(obs), min(N_DATASET_POINTS, len(obs)), replace=False)
    with torch.no_grad():
        m_data = margin(encoder(torch.from_numpy(obs[idx]))).numpy()

    plt.scatter(theta[idx][safe[idx]], m_data[safe[idx]], s=4, alpha=0.3, label="dataset: safe")
    plt.scatter(theta[idx][~safe[idx]], m_data[~safe[idx]], s=4, alpha=0.3, label="dataset: failed")
    plt.plot(angles, m_sweep, color="k", lw=1.5, label="sweep (ω = 0)")
    plt.axhline(0, color="k", lw=0.8)
    for b in (-FAIL_ANGLE, FAIL_ANGLE):
        plt.axvline(b, color="r", ls="--", lw=0.8)
    plt.xlabel("true angle θ (rad)")
    plt.ylabel("margin m(E(image))")
    plt.title("Learned margin vs. true failure boundary (red)")
    plt.legend()
    plt.show()
