from gym_env import CarlaEnv

env = CarlaEnv()

state = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Take random action
    state, reward, done, _ = env.step(action)
    if done:
        break

env.close()
