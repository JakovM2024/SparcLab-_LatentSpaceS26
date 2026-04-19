import gymnasium as gym
import numpy as np
import random
import math
from PIL import Image, ImageDraw

IMG_SIZE = 84
POLE_LEN = 55  # pixels

class Environment(gym.Env):
    metadata = {"render_modes" : "human", "render_fps": 4}

    def __init__(self, num_steps = 10000, render_mode = "human"):

        self.action_space = gym.spaces.Box(low = -10, high = 10, shape = (1,), dtype = np.float32)
        self.window_size = 512
        self.num_steps = num_steps
        self.curr_step = 0

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
        )

        rand_angle = random.uniform(-.1, .1)
        self.state = np.array([rand_angle, 0], dtype=np.float32)

    def render(self):
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx, cy = IMG_SIZE // 2, IMG_SIZE - 10
        angle = float(self.state[0])
        tip_x = cx + int(POLE_LEN * math.sin(angle))
        tip_y = cy - int(POLE_LEN * math.cos(angle))

        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(180, 180, 180))
        draw.line([cx, cy, tip_x, tip_y], fill=(255, 80, 80), width=10)
        draw.ellipse([tip_x - 10, tip_y - 10, tip_x + 10, tip_y + 10], fill=(255, 80, 80))

        return np.array(img, dtype=np.uint8)

    def _get_obs(self):
        return self.render()

    def reset(self, seed=None, options=None):
        rand_angle = random.uniform(-.1, .1)
        self.state = np.array([rand_angle, 0], dtype=np.float32)
        self.curr_step = 0
        return self._get_obs(), 0

    def step(self, action):
        angle, angular_vel = self.state

        gravity = 9.8
        mass = 5
        pole_length = 1
        delta_time = .02
        damping = .1

        angular_acceleration = ((gravity / pole_length) * math.sin(angle)) - action / (mass * pole_length**2) - \
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

        return self._get_obs(), reward, done, truncated, 0
