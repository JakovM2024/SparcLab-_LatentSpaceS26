import numpy as np
import torch
import matplotlib.pyplot as plt
from latent_encoder import Encoder, Decoder, Dynamics

NUM_EXAMPLES = 6
LATENT_SIZE = 16
ROLLOUT_STEPS = 10


def load_models():
    encoder = Encoder(latent_size=LATENT_SIZE)
    decoder = Decoder(latent_size=LATENT_SIZE)
    dynamics = Dynamics(latent_size=LATENT_SIZE)

    encoder.load_state_dict(torch.load("data/models/encoder.pt", weights_only=True))
    decoder.load_state_dict(torch.load("data/models/decoder.pt", weights_only=True))
    dynamics.load_state_dict(torch.load("data/models/dynamics.pt", weights_only=True))

    encoder.eval()
    decoder.eval()
    dynamics.eval()
    return encoder, decoder, dynamics


def sample_examples(encoder, decoder, dynamics):
    data = np.load("data/trajectories/demo_policy.npz")
    starts = data["episode_starts"]
    states = data["states"]
    actions = data["actions"]
    window = dynamics.window

    current_imgs, true_future_imgs, pred_future_imgs = [], [], []

    rng = np.random.default_rng()
    while len(current_imgs) < NUM_EXAMPLES:
        ep = rng.integers(0, len(starts) - 1)
        ep_states = states[starts[ep]:starts[ep + 1]]
        ep_actions = actions[starts[ep]:starts[ep + 1]]

        # need enough history for the window and enough future for the rollout
        if len(ep_states) <= window + ROLLOUT_STEPS:
            continue

        t = rng.integers(window, len(ep_states) - ROLLOUT_STEPS)

        with torch.no_grad():
            # encode the initial window of real frames ending at t
            z_window = encoder(torch.from_numpy(ep_states[t - window:t]))  # (window, latent)

            # roll dynamics forward ROLLOUT_STEPS using real actions
            for step in range(ROLLOUT_STEPS):
                action = torch.tensor(ep_actions[t + step], dtype=torch.float32).unsqueeze(0)  # (1, 1)
                z_pred = dynamics(z_window.unsqueeze(0), action).squeeze(0)  # (latent,)

                # slide window: drop oldest, append prediction
                z_window = torch.cat([z_window[1:], z_pred.unsqueeze(0)], dim=0)

            pred_img = decoder(z_pred.unsqueeze(0)).squeeze(0).numpy()  # (84, 84, 3)

        current_imgs.append(ep_states[t - 1])
        true_future_imgs.append(ep_states[t + ROLLOUT_STEPS - 1])
        pred_future_imgs.append(np.clip(pred_img, 0, 1))

    return current_imgs, true_future_imgs, pred_future_imgs


def show(current_imgs, true_future_imgs, pred_future_imgs):
    fig, axes = plt.subplots(3, NUM_EXAMPLES, figsize=(NUM_EXAMPLES * 2, 7))
    fig.suptitle(f"Dynamics rollout: {ROLLOUT_STEPS} steps ahead", fontsize=14, fontweight="bold")

    row_labels = ["Current (t)", f"True (t+{ROLLOUT_STEPS})", f"Predicted\n(t+{ROLLOUT_STEPS})"]
    all_rows = [current_imgs, true_future_imgs, pred_future_imgs]
    row_y_positions = [0.78, 0.46, 0.14]

    for row, (imgs, label) in enumerate(zip(all_rows, row_labels)):
        for col, img in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Example {col + 1}", fontsize=10)

        fig.text(0.08, row_y_positions[row], label, fontsize=11,
                 va="center", ha="right", fontweight="bold")

    plt.subplots_adjust(left=0.12, top=0.92, hspace=0.05, wspace=0.05)
    plt.show()


if __name__ == "__main__":
    encoder, decoder, dynamics = load_models()
    current_imgs, true_future_imgs, pred_future_imgs = sample_examples(encoder, decoder, dynamics)
    show(current_imgs, true_future_imgs, pred_future_imgs)
