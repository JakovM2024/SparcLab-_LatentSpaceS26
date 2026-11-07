import gymnasium as gym
import numpy as np
import math
from PIL import Image, ImageDraw

IMG_SIZE = 84
POLE_LEN = 55            # pixels
FAIL_ANGLE = 0.4         # |theta| > FAIL_ANGLE is the failure set
MAX_RENDER_ANGLE = 0.62  # past this the tip ball starts leaving the 84px frame


class Environment(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode="rgb_array"):
        # Single source of truth for the actuator limit. With |u| <= 10 the pole
        # cannot be held from rest past roughly 0.2 rad, which is what makes the
        # safety problem non-trivial.
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
        )
        self.render_mode = render_mode

        self.gravity = 9.8
        self.mass = 5
        self.pole_length = 1
        self.delta_time = 0.02
        self.damping = 0.1

        self.state = np.zeros(2, dtype=np.float32)

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

    def _get_info(self):
        theta, omega = self.state
        return {"theta": float(theta), "omega": float(omega), "failed": abs(theta) > FAIL_ANGLE}

    def reset(self, seed=None, options=None):
        """options: {"theta0": float, "omega0": float}. Defaults to upright at rest."""
        super().reset(seed=seed)
        options = options or {}
        self.state = np.array(
            [options.get("theta0", 0.0), options.get("omega0", 0.0)], dtype=np.float32
        )
        return self._get_obs(), self._get_info()

    def step(self, action):
        theta, omega = self.state
        # Actuator saturates here too, so the applied torque never exceeds the bound.
        u = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0],
                          self.action_space.low[0], self.action_space.high[0]))

        inertia = self.mass * self.pole_length ** 2
        # Convention: positive u is positive torque, so u = -(Kp*theta + Kd*omega) stabilizes.
        angular_acceleration = (self.gravity / self.pole_length) * math.sin(theta) \
                               + u / inertia \
                               - (self.damping / inertia) * omega
        omega = omega + angular_acceleration * self.delta_time
        theta = theta + omega * self.delta_time
        self.state = np.array([theta, omega], dtype=np.float32)

        info = self._get_info()
        terminated = info["failed"]   # failure flag; data collection keeps rolling past it
        reward = 0.0 if terminated else 0.5
        return self._get_obs(), reward, terminated, False, info
