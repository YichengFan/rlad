import gymnasium as gym
from gymnasium import spaces
import numpy as np
from carla_env_core import CarlaLaneKeepingEnv  # Import base environment

class CarlaGymEnv(gym.Env):  # Inherit from gymnasium.Env
    def __init__(self):
        super(CarlaGymEnv, self).__init__()
        self.env = CarlaLaneKeepingEnv()

        # Define action space: 0 = left, 1 = straight, 2 = right.
        self.action_space = spaces.Discrete(3)

        # Define observation space to match the unnormalized state values.
        self.observation_space = spaces.Box(
            low=np.array([-100.0, -180.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 180.0, 100.0, 50.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        """Gymnasium requires a return of (obs, info) on reset."""
        state = self.env.reset()
        info = {}
        return state, info

    def step(self, action):
        """Gymnasium requires a return of (obs, reward, done, truncated, info)."""
        state, reward, done = self.env.step(action)
        reward = float(reward)
        truncated = False  # No truncation conditions for now.
        info = {}
        return state, reward, done, truncated, info

    def render(self, mode="human"):
        # Optional: Add visualization if needed.
        pass

    def close(self):
        self.env.close()
