import gymnasium as gym
import numpy as np
import random
import math

class Environment(gym.Env):
    metadata = {"render_modes" : "human", "render_fps": 4}
    

    def __init__(self, num_steps = 10000, render_mode = "human"):
        
        self.action_space = gym.spaces.Box(low = -10, high = 10, shape = (1,), dtype = np.float32)
        self.window_size = 512
        self.num_steps = num_steps
        self.curr_step = 0

        #array holds these values, [pole angle, pole angular velocity
        low_bounds = np.array([-.5, -np.inf])
        high_bounds = np.array([.5, np.inf])

        self.observation_space = gym.spaces.Box(low = low_bounds, high = high_bounds, shape = (2,), dtype = np.float32) 

        
        rand_angle = random.uniform(-.1, .1)
        self.state = np.array([rand_angle , 0], dtype = np.float32)
        
        #[pole angle, pole angular velocity]
        #self._target_location = np.array([0, 0], dtype = np.float32) try learning without telling it to keep the pole upright

        #self.render_mode = self.metadata[render_mode]

    
    def _get_obs(self):
        return self.state
    
    def reset(self, seed = None, options = None):
        rand_angle = random.uniform(-.1, .1)
        self.state = np.array([rand_angle , 0], dtype = np.float32)
        self.curr_step = 0

        observation = self._get_obs()
        #self._render_frame()

        return observation, 0
    
    def step(self, action):
        angle, angular_vel = self.state

        gravity = 9.8
        mass = 5
        pole_length = 1
        delta_time = .02
        damping = .1

        angular_acceleration = ((gravity / pole_length) * math.sin(angle) ) - action/ (mass * pole_length**2) - \
                                (damping / (mass * pole_length**2)) * angular_vel 
        angular_vel = angular_vel + angular_acceleration * delta_time
        angle = angle + angular_vel * delta_time

        self.state = np.array([angle, angular_vel], dtype=np.float32)

        done = False
        reward = 0
        truncated = False

        if angle > .4 or angle < -.4:
            done = True
        
        else:
            reward += .5
        
        #if self.num_steps <= self.curr_step:
        #    truncated = True
        
        #self.curr_step += 1

        return self.state, reward, done, truncated, 0
        








