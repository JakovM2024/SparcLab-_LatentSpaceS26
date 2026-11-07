import os
import numpy as np
from enviornment import Environment, FAIL_ANGLE, MAX_RENDER_ANGLE

SEED = 0
N_EPISODES = 100
HORIZON = 200          # steps per episode (4 s at dt = 0.02)
THETA_STOP = 0.6       # keep rolling past failure until here; must stay below MAX_RENDER_ANGLE
THETA0_RANGE = 0.4     # initial angle sampled uniformly over the whole safe set
OMEGA0_RANGE = 1.5     # free fall from 0 to 0.4 rad reaches about 1.2 rad/s
PD_NOISE_STD = 3.0
OUT_DIR = "data/trajectories"

POLICY_TYPES = ["pd", "noisy_pd", "random", "pd_then_random"]

assert THETA_STOP < MAX_RENDER_ANGLE


def policy(env, policy_type, rng, step, switch_step):
    theta, omega = env.state
    low, high = env.action_space.low[0], env.action_space.high[0]
    pd = -(100 * theta + 32 * omega)

    if policy_type == "pd":
        u = pd
    elif policy_type == "noisy_pd":
        u = pd + rng.normal(0, PD_NOISE_STD)
    elif policy_type == "random":
        u = rng.uniform(low, high)
    elif policy_type == "pd_then_random":
        u = pd if step < switch_step else rng.uniform(low, high)
    else:
        raise ValueError(f"unknown policy type: {policy_type}")

    return np.float32(np.clip(u, low, high))


def get_data(policy_type, rng, n_episodes=N_EPISODES, horizon=HORIZON, out_dir=OUT_DIR):
    """Each episode stores T+1 frames and T actions:
        obs[k], theta[k], omega[k], safe[k]  for frames  k in [episode_starts[i], episode_starts[i+1])
        actions[j]                            for actions j in [action_starts[i],  action_starts[i+1])
    actions[action_starts[i] + t] moves frame episode_starts[i] + t to frame episode_starts[i] + t + 1.
    """
    env = Environment()
    obs_list, act_list, theta_list, omega_list = [], [], [], []
    episode_starts, action_starts = [0], [0]

    for _ in range(n_episodes):
        theta0 = rng.uniform(-THETA0_RANGE, THETA0_RANGE)
        omega0 = rng.uniform(-OMEGA0_RANGE, OMEGA0_RANGE)
        obs, info = env.reset(options={"theta0": theta0, "omega0": omega0})
        switch_step = rng.integers(0, horizon)

        for step in range(horizon):
            obs_list.append(obs)
            theta_list.append(info["theta"])
            omega_list.append(info["omega"])

            action = policy(env, policy_type, rng, step, switch_step)
            obs, _, _, _, info = env.step(action)
            act_list.append(action)

            if abs(info["theta"]) >= THETA_STOP:
                break

        # final frame of the episode
        obs_list.append(obs)
        theta_list.append(info["theta"])
        omega_list.append(info["omega"])

        episode_starts.append(len(obs_list))
        action_starts.append(len(act_list))

    theta = np.array(theta_list, dtype=np.float32)
    safe = np.abs(theta) <= FAIL_ANGLE   # label from the state, not from episode termination

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{policy_type}_policy.npz")
    np.savez_compressed(
        path,
        obs=np.array(obs_list, dtype=np.uint8),
        actions=np.array(act_list, dtype=np.float32).reshape(-1, 1),
        theta=theta,
        omega=np.array(omega_list, dtype=np.float32),
        safe=safe,
        episode_starts=np.array(episode_starts),
        action_starts=np.array(action_starts),
    )
    print(f"{policy_type:>15}: {len(obs_list)} frames, {(~safe).sum()} failed "
          f"({(~safe).mean():.1%}), saved to {path}")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    for policy_type in POLICY_TYPES:
        get_data(policy_type, rng)
