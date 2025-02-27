import numpy as np
from carla_env_core import CarlaLaneKeepingEnv
import random

env = CarlaLaneKeepingEnv(num_pedestrians=10)
state_size = 3  # [lateral_position, heading_angle, speed]
action_size = 3  # Left, Straight, Right

# Training loop
for episode in range(100):  # Adjust based on needs
    state = env.reset()
    total_reward = 0

    for step in range(500):  # Adjust for training length
        action = random.choice([-1, 0, 1])  # Dummy action selection (Replace with RL model)
        next_state, reward, done = env.step(action)
        total_reward += reward

        if done:
            break

    print(f"Episode {episode}: Total Reward = {total_reward}")

env.close()
