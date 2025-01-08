from carla_gym_env import CarlaEnv
from stable_baselines3 import PPO

# Initialize the Carla environment
env = CarlaEnv()

# Load the PPO model
model = PPO.load("ppo_carla")

# Test the PPO agent
obs = env.reset()
for _ in range(1000):  # Number of steps to test
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    if done:
        obs = env.reset()

env.close()
