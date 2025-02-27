from stable_baselines3 import DQN
from carla_env_wrapper import CarlaGymEnv
import time


def main():
    # Initialize the CARLA Gym environment
    env = CarlaGymEnv()

    # Load the trained model
    model_path = "lane_keeping_dqn_model.zip"
    model = DQN.load(model_path, env=env)
    print("Loaded trained model from", model_path)

    # Optionally run multiple episodes for evaluation
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"\nStarting evaluation episode {episode + 1}")
        obs, _ = env.reset()  # Reset environment
        done = False
        total_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)  # Predict action
            obs, reward, done, truncated, info = env.step(action)  # Apply action
            total_reward += reward

            # Print current step information
            print(f"Action: {action}, Reward: {reward}, Done: {done}")
            # Optionally, call env.render() here if implemented for visualization
            time.sleep(0.1)  # Slow down to observe behavior

            # Check for truncation as well as done
            if truncated:
                print("Episode truncated!")
                break

        print(f"Episode {episode + 1} completed with total reward: {total_reward}")

    print("Evaluation completed!")
    env.close()


if __name__ == "__main__":
    main()

