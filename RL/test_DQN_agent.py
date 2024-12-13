from gym_env import CarlaEnv
from stable_baselines3 import DQN

env = CarlaEnv()
model = DQN.load("dqn_carla")
obs = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()
