import gymnasium as gym
import numpy as np
import random

class Environment(gym.env):
    metadata = {"render_modes" : "human", "render_fps": 4}
    

    def __init__(self, render_mode = "human"):
        
        self.action_space = gym.spaces.Box(low = -3, high = 3, shape = (1,), dtype = np.float32)
        self.window_size = 512

        #array holds these values, [cart postion, cart velocity, pole angle, pole angular velocity
        low_bounds = np.array([-5, -np.inf, -30, -np.inf])
        high_bounds = np.array([5, np.inf, 30, np.inf])

        self.observation_space = gym.spaces.Box(low = low_bounds, high = high_bounds, shape = (4,), dtype = np.float32) 

        
        rand_angle = random.sample(range(-.1, .1), k = 1)
        self.state = np.array([0, 0, rand_angle , 0], dtype = np.float32)
        
        #[pole angle, pole angular velocity]
        self._target_location = np.array([0, 0], dtype = np.float32)

        self.render_mode = self.metadata["render_modes"]

    
    def _get_obs(self):
        return self.state
    
    def reset(self, seed = None, options = None):
        rand_angle = random.sample(range(-.1, .1), k = 1)
        self.state = np.array([0, 0, rand_angle , 0], dtype = np.float32)

        observation = self._get_obs()
        #self._render_frame()

        return observation
    
    def step(self, action):
        pos, velocity, angle, angular_vel = self.state

        gravity = -9.8
        mass_cart = 2
        mass_pole = .5
        pole_length = 1
        delta_time = .25

        new_pos = self._agent_location[0] + self._agent_location[1] * self.delta_time + action * (self.delta_time**2)






