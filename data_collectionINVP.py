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
        state, _ = env.reset()
        for step in range(5000):
            action = policy(state, policy_type)
            next_state, reward, done, truncated,___ = env.step(action)

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            safe_labels.append(not done)

            state = next_state

            if done or truncated:
                break

            
        episode_starts.append(len(states))

    ep1_states = states[episode_starts[0]:episode_starts[1]]

    print(ep1_states)
    print('\n')

    
##    np.savez("data/trajectories/" + policy_type +"_policy.npz",
  ##      states = np.array(states),
  ##      actions = np.array(actions),
  ##      next_states = np.array(next_states),
  ##      safe = np.array(safe_labels),
  ##      episode_starts = np.array(episode_starts)
  ##  )
    

def policy(state, policy_type):

    if policy_type == "safe":
        angle, angular_vel = state
        force = -(100 * angle + 32 * angular_vel)
        return np.clip(force, -60, 60)
    
    if policy_type == "unsafe":
        angle, angular_vel = state
        force = -10 * angle - 2 * angular_vel
        return np.clip(force, -10,10)
    
    #if policy_type == "random":

if __name__ == "__main__":
    get_data("safe")
    get_data("unsafe")



