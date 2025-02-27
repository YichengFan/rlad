from stable_baselines3 import DQN
from carla_env_wrapper import CarlaGymEnv
from stable_baselines3.common.logger import configure
import os
import datetime

def main():
    env = CarlaGymEnv()

    # Ensure TensorBoard Log Directory Exists
    log_dir = "./logs/tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    # Clear old logs before training starts
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Define model save path
    model_path = "lane_keeping_dqn_model.zip"

    # Check if a model exists to resume training
    if os.path.exists(model_path):
        model = DQN.load(model_path, env=env, device="auto")
        print("Loaded existing model from", model_path)
    else:
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [256, 256]},  # Better neural network
            learning_rate=0.0001,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=128,
            tau=0.01,
            gamma=0.99,
            train_freq=4,
            target_update_interval=500,
            exploration_fraction=0.6,
            exploration_final_eps=0.01,
            verbose=1,
            device="auto",
            tensorboard_log=log_dir,  # Enable TensorBoard Logging
        )
        print("Created a new model")

    # Set up the logger properly
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Train in multiple runs
    train_runs = 5       # Run training in 5 blocks
    steps_per_run = 20000  # Number of timesteps per block

    for i in range(train_runs):
        print(f"Training run {i+1} for {steps_per_run} timesteps...")
        model.learn(total_timesteps=steps_per_run, reset_num_timesteps=False, tb_log_name="DQN_training")
        model.save(model_path)
        print(f"Model saved after training run {i+1}")

if __name__ == "__main__":
    main()
