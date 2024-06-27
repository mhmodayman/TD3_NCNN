from config.config import *

ITERS = 1000  # number of episodes
learning_rate = 0.001
batch_size = 256
buffer_size = 300_000
total_time_steps = 100
learning_starts = 10_000
gamma = 0.98
tau = 0.02
train_freq = 8
gradient_steps = 8

params['max_time_episode'] = total_time_steps

env = gym.make('carla-v0', params=params)
obs = env.reset()

model_name = 'TD3'
model = TD3('MlpPolicy', env,  # MlpPolicy, MultiInputPolicy
            verbose=1,
            learning_rate=learning_rate,
            tensorboard_log=current_session_logdir,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            gamma=gamma,
            tau=tau,
            train_freq=train_freq,
            gradient_steps=gradient_steps,)

iteration = 0
while iteration < ITERS:
    iteration += 1

    print('Iteration ', iteration, ' is to commence...')

    model.learn(total_timesteps=total_time_steps, reset_num_timesteps=False, tb_log_name=model_name)
    model.save(f"{current_session_models_dir}/{total_time_steps * iteration}")

    print('Iteration ', iteration, ' has been trained')
