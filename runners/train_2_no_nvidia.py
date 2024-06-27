print('This is training without nvidia, change path in the variable (record_saving_path) located in config.py to appropriate one first')
print('This is training without nvidia, change path in the variable (record_saving_path) located in config.py to appropriate one first')
print('This is training without nvidia, change path in the variable (record_saving_path) located in config.py to appropriate one first')
print('This is training without nvidia, change path in the variable (record_saving_path) located in config.py to appropriate one first')
print('This is training without nvidia, change path in the variable (record_saving_path) located in config.py to appropriate one first')
print('This is training without nvidia, change path in the variable (record_saving_path) located in config.py to appropriate one first')
print('This is training without nvidia, change path in the variable (record_saving_path) located in config.py to appropriate one first')

import os
import sys
sys.path.append(os.getcwd())
from modules_importer import *

import numpy as np
import matplotlib.pyplot as plt
from models.TD3 import utils
from models.TD3.TD3 import Agent
from keras.models import load_model
from config.config import *

# import random
# # set seeds
# random_seed = 1
# env.seed(random_seed)
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# random.seed(random_seed)

divisor = 10
max_episodes = divisor * 400
max_timesteps = divisor * 200
# learning starts after 2000 timesteps pass in total across episodes
# i.e., it doesn't reset after each episode
learning_starts = 2 * max_timesteps
params['max_time_episode'] = max_timesteps
env = gym.make('carla-v0', params=params)

# Set exploration noise for calculating action based on some noise factor
exploration_noise = 0.1

# Define observation and action space
state_space = np.prod(env.observation_space.shape)  # B x C x H x W  (B is batch size)
action_space_shape = env.action_space.shape[0]
action_space = env.action_space
max_action = float(env.action_space.high[0])

try:
    # Create Agent
    policy = Agent(state_space, action_space_shape, max_action)
except:
    print('change init_weights(self, layer, gain=1) from case 2 to case 1 '
          'until you have a saved checkpoint, '
          'then you can switch again to case 2')
    sys.exit()

current_checkpoint_number = 0
if last_checkpoint_number:
    try:
        policy.load(last_checkpoint_number)
        current_checkpoint_number = int(last_checkpoint_number) + 1
        print(f"loaded checkpoint {last_checkpoint_number}")
    except:
        raise IOError(f"Couldn't load policy {last_checkpoint_number}")

try:
    policy.load("final")  # final checkpoint is different from last checkpoint
except:
    pass
    # raise IOError("Couldn't load policy")

# Create Replay Buffer
replay_buffer = utils.ReplayBuffer()

# Train the model
ep_reward = []  # get list of reward for range(max_episodes)

max_time_step_reached_in_any_episode = 0
for episode in range(1, max_episodes + 1):
    avg_reward = 0
    state = env.reset()

    for t in range(1, max_timesteps + 1):
        # --------------------------------------------------------------------------------------------------------------
        # case 1
        # select action and add exploration noise:
        action = policy.select_action(state, action_space) + np.random.normal(0, max_action * exploration_noise, size=action_space_shape)
        # --------------------------------------------------------------------------------------------------------------
        # # --------------------------------------------------------------------------------------------------------------
        # # case 2
        # action = nvidia_model.predict(state.reshape((1, 66, 200, -1)), verbose=0)[0]
        # # --------------------------------------------------------------------------------------------------------------

        # take action in env:
        next_state, reward, done, _ = env.step(action, 2)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        avg_reward += reward

        if len(replay_buffer) > learning_starts:  # make sure sample is less than overall population
            policy.train(replay_buffer)  # training mode

        # if episode is done then update policy:
        if done or t >= max_timesteps:
            ep_reward.append(avg_reward)
            print(f"Episode {episode} reward: {avg_reward} | Rolling average: {np.mean(ep_reward)}")
            if t > max_time_step_reached_in_any_episode:
                max_time_step_reached_in_any_episode = t
            print(f"                   max_time_step_reached_in_any_episode: {max_time_step_reached_in_any_episode}")
            break

    if np.mean(ep_reward[-10:]) >= 300:
        policy.save("final")
        break

    if episode % divisor == 0 and episode > 0:
        # Save policy and optimizer every divisor episodes
        policy.save(str(checkpoint_name_format % current_checkpoint_number))
        current_checkpoint_number += 1

env.close()

# Display Scores
fig = plt.figure()
plt.plot(np.arange(1, len(ep_reward) + 1), ep_reward)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
