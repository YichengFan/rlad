from carla_env_wrapper import CarlaGymEnv

env = CarlaGymEnv()
state = env.reset()
print("Initial state:", state)

