from config.config import *
from misc.retrieve_last_saved_model_path import retrieve_lsat_saved_model_path

ITERS = 1000  # number of episodes
learning_rate = 0.001
batch_size = 16
some_int_factor = 30
buffer_size = some_int_factor * batch_size
total_time_steps = 500

params['max_time_episode'] = total_time_steps

model_name = 'TD3'
last_saved_model_path = retrieve_lsat_saved_model_path(models_dir, total_time_steps)

env = gym.make('carla-v0', params=params)
obs = env.reset()

model = TD3.load(os.path.join(os.getcwd(), last_saved_model_path), env,
                 verbose=1,
                 learning_rate=learning_rate,
                 tensorboard_log=current_session_logdir,
                 batch_size=batch_size,
                 buffer_size=buffer_size,
                 learning_starts=total_time_steps*2)

iteration = 0
while iteration < ITERS:
    iteration += 1

    print('Iteration ', iteration, ' is to commence...')

    model.learn(total_timesteps=total_time_steps, reset_num_timesteps=False, tb_log_name=model_name)
    model.save(f"{current_session_models_dir}/{total_time_steps * iteration}")

    print('Iteration ', iteration, ' has been trained')
