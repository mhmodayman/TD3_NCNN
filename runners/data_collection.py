from config.config import *
import pygame

total_timesteps = 100_000
params['max_time_episode'] = total_timesteps
params['enable_keyboard'] = True

env = gym.make('carla-v0', params=params)
obs = env.reset()

iteration = 0

clock = pygame.time.Clock()
while True:
    # NO model learning

    clock.tick_busy_loop(60)

    iteration += 1

    # print('Iteration ', iteration, ' is to commence...')

    Quit, _, done, info = env.record_step(clock)
    if Quit:
        obs = env.reset()
        pygame.quit()
        break

    if done:
        obs = env.reset()

    # print('Iteration ', iteration, ' has been trained')
sys.exit()
