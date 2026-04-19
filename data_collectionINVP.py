from enviornment import Environment
import numpy as np


def get_data(policy_type):
    env = Environment()
    states = []
    actions = []
    next_states = []
    safe_labels = []
    episode_starts = [0]

    for episode in range(100):
        obs, _ = env.reset()
        for step in range(5000):
            action = policy(env.state, policy_type)
            next_obs, reward, done, truncated, ___ = env.step(action)

            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            safe_labels.append(not done)

            obs = next_obs

            if done or truncated:
                break

        episode_starts.append(len(states))

    np.savez("data/trajectories/" + policy_type + "_policy.npz",
        states = np.array(states, dtype=np.uint8),
        actions = np.array(actions),
        next_states = np.array(next_states, dtype=np.uint8),
        safe = np.array(safe_labels),
        episode_starts = np.array(episode_starts)
    )


def policy(state, policy_type):

    if policy_type == "safe":
        angle, angular_vel = state
        force = -(100 * angle + 32 * angular_vel)
        return np.clip(force, -60, 60)

    if policy_type == "unsafe":
        angle, angular_vel = state
        force = -10 * angle - 2 * angular_vel
        return np.clip(force, -10, 10)

if __name__ == "__main__":
    get_data("safe")
    get_data("unsafe")
