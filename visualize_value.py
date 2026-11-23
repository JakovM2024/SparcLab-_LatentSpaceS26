"""Compare the learned safety value V(z) = Q(z, pi_safe(z)) against the ground-truth safe set.

For each grid point (theta, omega) we render WINDOW frames at theta - k * omega * dt (k = WINDOW-1..0),
encode them into a latent window, and evaluate V and pi_safe. The ground truth comes from
ground_truth.py, which solves the problem exactly on the true state.
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from enviornment import Environment, FAIL_ANGLE
from latent_encoder import Encoder, WINDOW
from value_function import Actor, load_critic, safety_value
from ground_truth import viability_grid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_SIZE = 16


def load(module, path):
    module.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))
    return module.to(DEVICE).eval()


@torch.no_grad()
def learned_grid(encoder, actor, critic, thetas, omegas):
    env = Environment()
    dt = env.delta_time
    frames = []
    for th in thetas:
        for om in omegas:
            for k in range(WINDOW - 1, -1, -1):          # oldest frame first
                env.state = np.array([th - k * om * dt, om], dtype=np.float32)
                frames.append(env.render())
    frames = np.stack(frames)                             # (n_th * n_om * WINDOW, 84, 84, 3)

    z = torch.cat([encoder(torch.from_numpy(frames[b:b + 768]).to(DEVICE))
                   for b in range(0, len(frames), 768)])
    w = z.view(len(thetas) * len(omegas), WINDOW, -1)
    V = safety_value(actor, critic, w).cpu().numpy().reshape(len(thetas), len(omegas))
    pi = actor(w).cpu().numpy().reshape(len(thetas), len(omegas))
    return V, pi


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="data/models/value", help="loads <prefix>_actor.pt / _critic.pt")
    p.add_argument("--n", type=int, default=61, help="grid points per axis")
    p.add_argument("--omega-max", type=float, default=3.0)
    p.add_argument("--save", default=None, help="optional path to save the figure")
    args = p.parse_args()

    encoder = load(Encoder(latent_size=LATENT_SIZE), "data/models/encoder.pt")
    actor = load(Actor(LATENT_SIZE), f"{args.prefix}_actor.pt")
    critic = load_critic(f"{args.prefix}_critic.pt", LATENT_SIZE, DEVICE)

    thetas = np.linspace(-0.6, 0.6, args.n)
    omegas = np.linspace(-args.omega_max, args.omega_max, args.n)
    V, pi = learned_grid(encoder, actor, critic, thetas, omegas)

    gt_th, gt_om, V_gt = viability_grid(theta_max=0.6, omega_max=args.omega_max)
    # ground truth sampled at the learned grid points (nearest neighbor)
    ii = np.abs(gt_th[:, None] - thetas[None, :]).argmin(0)
    jj = np.abs(gt_om[:, None] - omegas[None, :]).argmin(0)
    gt_safe = V_gt[np.ix_(ii, jj)] >= 0

    inside = np.abs(thetas)[:, None] <= FAIL_ANGLE
    agree = (V >= 0) == gt_safe
    fp = np.mean((V >= 0) & ~gt_safe)   # learned says safe, truly doomed: the dangerous error
    fn = np.mean((V < 0) & gt_safe)
    print(f"Sign agreement with ground truth: {agree.mean():.3f} overall, {agree[np.broadcast_to(inside, agree.shape)].mean():.3f} "
          f"inside |theta| <= {FAIL_ANGLE}   (unsafe-called-safe {fp:.3f}, safe-called-unsafe {fn:.3f})")

    ext = [omegas[0], omegas[-1], thetas[0], thetas[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmax = np.abs(V).max()

    im = axes[0].imshow(V, origin="lower", extent=ext, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[0].contour(omegas, thetas, V, levels=[0], colors="k", linewidths=1.5)
    axes[0].contour(gt_om, gt_th, V_gt, levels=[0], colors="lime", linewidths=1.5, linestyles="--")
    axes[0].set_title("Learned V (black: V=0)  vs  ground truth (green dashed)")
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(agree, origin="lower", extent=ext, aspect="auto", cmap="Greys_r", vmin=0, vmax=1)
    axes[1].set_title(f"Sign agreement (white = agree), {agree.mean():.1%}")

    im = axes[2].imshow(pi, origin="lower", extent=ext, aspect="auto", cmap="coolwarm")
    axes[2].contour(gt_om, gt_th, V_gt, levels=[0], colors="lime", linewidths=1.5, linestyles="--")
    axes[2].set_title("Safety policy pi_safe (action)")
    fig.colorbar(im, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("omega (rad/s)")
        ax.set_ylabel("theta (rad)")
        for b in (-FAIL_ANGLE, FAIL_ANGLE):
            ax.axhline(b, color="r", ls=":", lw=0.8)
    plt.tight_layout()
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=120)
        print(f"Saved {args.save}")
    plt.show()
