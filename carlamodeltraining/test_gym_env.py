from stable_baselines3.common.env_checker import check_env
from carla_env_wrapper import CarlaGymEnv

# Create an instance of the Carla Gym environment
env = CarlaGymEnv()

# Check if the environment is compatible with Gym
check_env(env, warn=True)
print("Environment is compatible with Gym!")
