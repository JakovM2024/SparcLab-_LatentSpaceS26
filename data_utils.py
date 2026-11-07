import numpy as np
from data_collectionINVP import POLICY_TYPES

DATA_DIR = "data/trajectories"


def load_split(names=POLICY_TYPES, val_fraction=0.1, seed=0, keys=("obs",), data_dir=DATA_DIR):
    """Load the policy datasets and split them into train / val by WHOLE EPISODES.

    Consecutive frames of one episode are nearly identical, so a random frame-level split
    would leak training frames into validation. Splitting by episode avoids that.

    keys must be per-frame arrays (obs, theta, omega, safe), not actions.
    Returns (train, val): dicts of concatenated arrays plus "episode_starts" (frame indices
    into the concatenated arrays, length n_episodes + 1).
    """
    rng = np.random.default_rng(seed)
    parts = {"train": {k: [] for k in keys}, "val": {k: [] for k in keys}}
    starts = {"train": [0], "val": [0]}
    template = {}

    for name in names:
        data = np.load(f"{data_dir}/{name}_policy.npz")
        fs = data["episode_starts"]
        arrays = {k: data[k] for k in keys}  # decompress each key once
        template = {k: v[:0] for k, v in arrays.items()}

        n_ep = len(fs) - 1
        n_val = int(round(val_fraction * n_ep))
        val_eps = set(rng.choice(n_ep, n_val, replace=False).tolist()) if n_val > 0 else set()

        for i in range(n_ep):
            split = "val" if i in val_eps else "train"
            for k in keys:
                parts[split][k].append(arrays[k][fs[i]:fs[i + 1]])
            starts[split].append(starts[split][-1] + (fs[i + 1] - fs[i]))

    out = []
    for split in ("train", "val"):
        d = {k: (np.concatenate(v) if v else template[k]) for k, v in parts[split].items()}
        d["episode_starts"] = np.array(starts[split])
        out.append(d)
    return out[0], out[1]
