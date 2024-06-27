from config.config import *

total_timesteps = 1000
params['max_time_episode'] = total_timesteps

env = gym.make('carla-v0', params=params)
obs = env.reset()

iteration = 0
while True:
    # NO model learning

    iteration += 1

    print('Iteration ', iteration, ' is to commence...')
    action = [0.0, 0.0]
    obs, r, done, info = env.step(action, 1)

    if done:
        obs = env.reset()
    print('Iteration ', iteration, ' has been trained')
