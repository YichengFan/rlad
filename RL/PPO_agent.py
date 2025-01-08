from carla_gym_env import CarlaEnv  # Import your custom Gym environment
from stable_baselines3 import PPO  # Import PPO

# Initialize the Carla environment
env = CarlaEnv()

# Initialize the PPO model
model = PPO(
    policy="CnnPolicy",           # Use a CNN-based policy for image inputs
    env=env,                      # Pass the Carla environment
    learning_rate=3e-4,           # Learning rate for optimization
    n_steps=2048,                 # Number of steps to run for each update
    batch_size=64,                # Minibatch size for updates
    n_epochs=10,                  # Number of training epochs per update
    gamma=0.99,                   # Discount factor
    gae_lambda=0.95,              # Lambda for GAE (advantage estimation)
    clip_range=0.2,               # Clipping range for PPO
    verbose=1,                    # Print training logs
)

# Train the PPO model
model.learn(total_timesteps=50000)  # Adjust the timesteps as needed

# Save the PPO model
model.save("ppo_carla")

# Evaluate the PPO model
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward (PPO): {mean_reward}, Std Reward (PPO): {std_reward}")
