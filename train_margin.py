import argparse
import json
import os
import numpy as np
import torch
import torch.optim as optim
from latent_encoder import Encoder, Margin
from data_utils import load_split

DEVICE = torch.device("cude" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
FAIL_BAND = (0.35, 0.45)  # |theta| range near the boundary, where errors concentrate


def parse_args():
    p = argparse.ArgumentParser(description="Train the latent margin function l(z).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-hinge", type=float, default=1.0, help="hinge / sign loss at +-1")
    p.add_argument("--lambda-rank", type=float, default=0.0,
                   help="WGAN ranking term E_fail[m] - E_safe[m] (paper eq. 8); 0 disables")
    p.add_argument("--lambda-gp", type=float, default=1.0, help="gradient penalty weight")
    p.add_argument("--beta", type=float, default=1.0, help="target gradient norm in the penalty")
    p.add_argument("--one-sided-gp", action="store_true",
                   help="penalize only ||grad|| > beta (bounds the slope instead of fixing it)")
    p.add_argument("--bound", type=float, default=2.0, help="margin output lies in (-bound, bound)")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/models/margin.pt")
    p.add_argument("--run-name", default=None,
                   help="if set, write config + final metrics to data/runs/<run-name>.json")
    return p.parse_args()


def encode(encoder, images):
    with torch.no_grad():
        return torch.cat([
            encoder(torch.from_numpy(images[b:b + BATCH_SIZE]).to(DEVICE))
            for b in range(0, len(images), BATCH_SIZE)
        ])


def gradient_penalty(margin, z_safe, z_fail, beta, one_sided):
    # z_hat: random points between safe/failed latent pairs, where the boundary lives
    alpha = torch.rand(len(z_safe), 1,deivce = z_safe.device)
    z_hat = (alpha * z_safe + (1 - alpha) * z_fail).requires_grad_(True)
    grad = torch.autograd.grad(margin(z_hat).sum(), z_hat, create_graph=True)[0]
    gap = grad.norm(dim=1) - beta
    if one_sided:
        gap = torch.relu(gap)
    return (gap ** 2).mean()


def margin_loss(margin, z_safe, z_fail, args):
    """lambda_hinge * hinge  +  lambda_rank * (E_fail[m] - E_safe[m])  +  lambda_gp * GP."""
    m_safe, m_fail = margin(z_safe), margin(z_fail)
    hinge = torch.relu(1 - m_safe).mean() + torch.relu(1 + m_fail).mean()
    rank = m_fail.mean() - m_safe.mean()
    gp = gradient_penalty(margin, z_safe, z_fail, args.beta, args.one_sided_gp)
    total = args.lambda_hinge * hinge + args.lambda_rank * rank + args.lambda_gp * gp
    return total, {"hinge": hinge.item(), "rank": rank.item(), "gp": gp.item()}


@torch.no_grad()
def boundary_distances(z_safe, z_fail, max_points=4000, seed=0):
    """For each failed latent, distance to the nearest safe latent. With hinge targets +-1
    and slope ~beta, both can only hold if this distance is at least about 2 / beta."""
    g = torch.Generator().manual_seed(seed)
    zs = z_safe[torch.randperm(len(z_safe), generator=g)[:max_points]]
    zf = z_fail[torch.randperm(len(z_fail), generator=g)[:max_points]]
    d = torch.cdist(zf, zs).min(dim=1).values.cpu().numpy()
    return {q: float(np.quantile(d, q)) for q in (0.0, 0.1, 0.5, 0.9)}


@torch.no_grad()
def evaluate(margin, z, safe, theta, episode_starts):
    """Confusion matrix (TP = predicted safe and truly safe, as in the paper), accuracy in the
    band around the boundary, and smoothness: max |m(z_{t+1}) - m(z_t)| per episode."""
    m = margin(z).cpu().numpy()
    pred = m > 0
    out = {
        "TP": float(np.mean(pred & safe)), "TN": float(np.mean(~pred & ~safe)),
        "FP": float(np.mean(pred & ~safe)), "FN": float(np.mean(~pred & safe)),
    }
    out["acc"] = out["TP"] + out["TN"]
    band = (np.abs(theta) >= FAIL_BAND[0]) & (np.abs(theta) <= FAIL_BAND[1])
    out["band_acc"] = float(np.mean(pred[band] == safe[band])) if band.any() else float("nan")

    max_deltas = [np.abs(np.diff(m[s:e])).max()
                  for s, e in zip(episode_starts[:-1], episode_starts[1:]) if e - s > 1]
    out["max_dm_mean"] = float(np.mean(max_deltas))
    out["max_dm_std"] = float(np.std(max_deltas))
    return out


def fmt(e):
    return (f"acc {e['acc']:.3f}  band acc {e['band_acc']:.3f}  "
            f"TP {e['TP']:.3f} TN {e['TN']:.3f} FP {e['FP']:.3f} FN {e['FN']:.3f}  "
            f"max|dm| {e['max_dm_mean']:.3f}±{e['max_dm_std']:.3f}")


if __name__ == "__main__":
    args = parse_args()
    assert args.bound > 1, "bound must exceed the hinge targets (+-1)"
    torch.manual_seed(args.seed)

    encoder = Encoder(latent_size=16).to(DEVICE)
    encoder.load_state_dict(torch.load("data/models/encoder.pt", weights_only=True, map_location=DEVICE))
    encoder.eval()

    keys = ("obs", "safe", "theta")
    train, val = load_split(val_fraction=args.val_fraction, seed=args.seed, keys=keys)
    if len(val["obs"]) == 0:
        print("No validation set: evaluating on the training data instead.")
        val = train

    print("Encoding frames...")
    z_train, z_val = encode(encoder, train["obs"]), encode(encoder, val["obs"])
    safe_train = torch.from_numpy(train["safe"]).to(DEVICE)
    z_safe, z_fail = z_train[safe_train], z_train[~safe_train]
    print(f"train: {len(z_safe)} safe / {len(z_fail)} failed latents   val: {len(z_val)} frames")

    d = boundary_distances(z_safe, z_fail, seed=args.seed)
    print("nearest-safe distance from failed latents: "
          + "  ".join(f"q{int(q * 100)}={v:.3f}" for q, v in d.items())
          + f"   (hinge at +-1 with slope beta={args.beta} needs >= {2 / args.beta:.2f})")

    margin = Margin(latent_size=16, bound=args.bound).to(DEVICE)
    optimizer = optim.Adam(margin.parameters(), lr=args.lr)
    steps_per_epoch = max(1, len(z_safe) // BATCH_SIZE)

    for epoch in range(args.epochs):
        sums = {"total": 0.0, "hinge": 0.0, "rank": 0.0, "gp": 0.0}
        for _ in range(steps_per_epoch):
            # sample each class separately: failed frames are a small fraction of the data
            s = z_safe[torch.randint(len(z_safe), (BATCH_SIZE,))]
            u = z_fail[torch.randint(len(z_fail), (BATCH_SIZE,))]
            loss, parts = margin_loss(margin, s, u, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums["total"] += loss.item()
            for k, v in parts.items():
                sums[k] += v

        avg = {k: v / steps_per_epoch for k, v in sums.items()}
        ev = evaluate(margin, z_val, val["safe"], val["theta"], val["episode_starts"])
        print(f"Epoch {epoch + 1}/{args.epochs}  loss {avg['total']:.4f} "
              f"(hinge {avg['hinge']:.3f} rank {avg['rank']:.3f} gp {avg['gp']:.3f})  | val {fmt(ev)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(margin.state_dict(), args.out)
    print(f"Margin saved to {args.out}")

    if args.run_name:
        os.makedirs("data/runs", exist_ok=True)
        path = f"data/runs/{args.run_name}.json"
        with open(path, "w") as f:
            json.dump({"config": vars(args), "boundary_distances": d, "final_loss": avg,
                       "val": ev}, f, indent=2)
        print(f"Metrics written to {path}")
