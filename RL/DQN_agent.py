from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from gym_env import CarlaEnv  # Import your Gym environment

# Wrap the environment
env = CarlaEnv()
model = DQN("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("dqn_carla")


# Initialize the DQN model
model = DQN(
    policy="CnnPolicy",           # Use a CNN to process image data
    env=env,                      # Pass the Carla environment
    learning_rate=1e-4,           # Learning rate
    buffer_size=50000,            # Replay buffer size
    learning_starts=1000,         # Start learning after collecting enough data
    batch_size=32,                # Batch size for training
    tau=0.1,                      # Target network update rate
    gamma=0.99,                   # Discount factor
    train_freq=4,                 # Train every 4 steps
    target_update_interval=1000,  # Update the target network every 1000 steps
    verbose=1,                    # Print training logs
)

# Train the model
model.learn(total_timesteps=50000)  # Adjust timesteps as needed

# Save the model
model.save("dqn_carla")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# CNN Policy: The CnnPolicy processes the camera image as input.
# Discrete Actions: The DQN algorithm is suitable for discrete action spaces.
# Training Time: Training can take some time depending on the complexity of the task and hardware.