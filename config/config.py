import os
import sys
import gym
import gym_carla  # although faded, it is used in: env = gym.make('carla-v0', params=params)
import time
from pathlib import Path
from misc.open_yaml import open_yaml
from misc.retrieve_last_saved_checkpoint_path import retrieve_last_saved_checkpoint_path
# from stable_baselines3 import PPO, TD3, DDPG
import torch
from tensorflow.python.platform import build_info as tf_build_info
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)])
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

torch.cuda.empty_cache()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('torch.backends.cudnn.m.is_available()', torch.backends.cudnn.m.is_available())
print(tf_build_info.build_info)
print("tf_build_info.build_info['cuda_version']", tf_build_info.build_info['cuda_version'])
print("tf_build_info.build_info['cudnn_version']", tf_build_info.build_info['cudnn_version'])


repo_dir = str(Path(__file__).parent.parent)

# submodules -----------------------------------------------------------------------------------------------------------
sys.path.append(repo_dir + '/submodules/carla_base/PythonAPI/carla')

# YAML files -----------------------------------------------------------------------------------------------------------
rp_path = repo_dir + '/config/global_route_start_end.yaml'
rp_config = open_yaml(rp_path)
# ----------------------------------------------------------------------------------------------------------------------
records = [repo_dir + '/saved/record_dxdy',
           repo_dir + '/saved/record_all_1',
           repo_dir + '/saved/record_all_2',
           repo_dir + '/saved/record_all_3',
           repo_dir + '/saved/record_all_combined']

record_saving_path = records[-1]
if not os.path.exists(record_saving_path):
     os.makedirs(record_saving_path)

if not os.path.exists(record_saving_path + '/img'):
     os.makedirs(record_saving_path + '/img')
# ----------------------------------------------------------------------------------------------------------------------

visualize_logs_on_tensorboard = False

DISPLAY_MEANINGLESS_MULTIPLIER = 1  # used for simultaneous adjust of camera and lidar windows sizes, the bigger the bigger

params = {
    'origin': rp_config['origin'],  # origin of global route planner
    'destination': rp_config['destination'],  # destination of global route planner
    'spectator_pose': rp_config['spectator_pose'],
    'target_speed': 1.0,  # target speed (m/s)
    'wheelbase': 2.4,  # [m] Wheelbase of the audi a2 vehicle
    'controller_type': 'pure_pursuit',  # pure_pursuit, stanley, linear mpc
    'let_other_vehicles_move': False,
    'enable_lidar_observation': False,
    # ------------------------------------------------------------------------------------------------------------------
    # if keyboard is enabled, pygame rendering must be true
    'rendering':       True,
    'enable_keyboard': False,  # enabling keyboard will override autopilot behavior even if enabled
    # ------------------------------------------------------------------------------------------------------------------
    'enable_autopilot': False,
    # ------------------------------------------------------------------------------------------------------------------
    'random_ego_spawn': False,
    'number_of_vehicles': 2,
    'number_of_walkers': 0,
    # ------------------------------------------------------------------------------------------------------------------
    'height':               66     * DISPLAY_MEANINGLESS_MULTIPLIER,  # pygame camera window's height
    'width':               200     * DISPLAY_MEANINGLESS_MULTIPLIER,  # pygame camera window's width
    # ------------------------------------------------------------------------------------------------------------------
    'dt': 0.1,  # time interval between two frames
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.200 / DISPLAY_MEANINGLESS_MULTIPLIER,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'max_time_episode': 0,  # maximum timesteps per episode, 0 because runner will set it
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'town': 'Town03',  # which town to simulate
    'ego_vehicle_filter': 'vehicle.audi*',  # filter for defining ego vehicle audi.a2[0]
}

current_session_time = int(time.time())

# ----------------------------------------------------------------------------------------------------------------------
# SB3 related stuff
# namee = f"{current_session_time}_{params['height']}x{params['width']}/"
#
# models_dir = repo_dir + "/models/"
# current_session_models_dir = os.path.join(models_dir, namee)
#
# logdir = repo_dir + "/logs/"
# current_session_logdir = os.path.join(logdir, namee)
#
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)
# os.makedirs(current_session_models_dir)
#
# if not os.path.exists(logdir):
#     os.makedirs(logdir)
# os.makedirs(current_session_logdir)
# ----------------------------------------------------------------------------------------------------------------------

dir1 = record_saving_path + '/'
dir2 = dir1 + "checkpoints/"
dir3 = dir1 + 'checkpoints_optim/'
dir4 = dir2 + f"{current_session_time}_{params['height']}x{params['width']}/"
dir5 = dir3 + f"{current_session_time}_{params['height']}x{params['width']}/"

if not os.path.exists(dir1):
    os.makedirs(dir1)
if not os.path.exists(dir2):
    os.makedirs(dir2)
if not os.path.exists(dir3):
    os.makedirs(dir3)

checkpoint_name_format = "%03d"
active_dir, last_checkpoint_number1 = retrieve_last_saved_checkpoint_path(dir2)
active_dir_optim, last_checkpoint_number2 = retrieve_last_saved_checkpoint_path(dir3)
assert last_checkpoint_number1 == last_checkpoint_number2
last_checkpoint_number = last_checkpoint_number1

if not active_dir and not active_dir_optim:
    os.makedirs(dir4)
    os.makedirs(dir5)
    active_dir = dir4
    active_dir_optim = dir5
