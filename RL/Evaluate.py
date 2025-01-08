from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from RL.gym_env import env

# Load the trained model
model = DQN.load("dqn_carla")

# Evaluate on the environment
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()  # Add this if the environment supports rendering
    if done:
        obs = env.reset()

# Learning Curves
logger = configure("./logs", ["stdout", "tensorboard"])
model.set_logger(logger)
model.learn(total_timesteps=50000)

