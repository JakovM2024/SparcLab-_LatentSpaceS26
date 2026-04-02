import numpy as np

safe_data = np.load("data/trajectories/safe_policy.npz")
unsafe_data = np.load("data/trajectories/unsafe_policy.npz")

starts = safe_data["episode_starts"]
ep1_states = safe_data["safe"][starts[0]:starts[1]]

print(ep1_states)