from config.config import *
from misc.retrieve_last_saved_model_path import retrieve_lsat_saved_model_path

ITERS = 4  # number of episodes
default_batch_size = 64
default_n_envs = 1
some_int_factor = 10
buffer_size = some_int_factor * default_batch_size
n_steps = int(some_int_factor * default_batch_size / default_n_envs)  # is called train_freq is some models
total_timesteps = ITERS * n_steps

params['max_time_episode'] = n_steps

model_name = 'PPO'
last_saved_model_path = retrieve_lsat_saved_model_path(models_dir, total_timesteps)

env = gym.make('carla-v0', params=params)
obs = env.reset()

model = PPO.load(os.path.join(os.getcwd(), last_saved_model_path), env,
                 verbose=1,
                 learning_rate=0.001,
                 tensorboard_log=current_session_logdir,
                 n_steps=n_steps)

iteration = 0
while iteration < ITERS:
    iteration += 1

    print('Iteration ', iteration, ' is to commence...')

    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, tb_log_name=model_name)
    model.save(f"{current_session_models_dir}/{total_timesteps * iteration}")

    print('Iteration ', iteration, ' has been trained')
