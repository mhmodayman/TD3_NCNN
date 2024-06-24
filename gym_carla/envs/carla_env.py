import glob
import os
import sys
import time
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
repo_dir = str(Path(__file__).parent.parent.parent)
try:
    sys.path.append(glob.glob(repo_dir + '/submodules/carla_base/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import random
from gym import spaces
from gym_carla.envs.helper import *
from misc.keyboard_control import KeyboardControl
from controllers.pure_pursuit_controller import PurePursuitController
from controllers.stanley_controller import StanleyController
from controllers.linear_mpc_controller import LinearMPCController
# from agents.navigation.controller import VehiclePIDController


class CarlaEnv(gym.Env):
    def __init__(self, params):
        super(CarlaEnv, self).__init__()
        self.let_other_vehicles_move = params['let_other_vehicles_move']
        self.enable_autopilot = params['enable_autopilot']
        # --------------------------------------------------------------------------------------------------------------
        # if keyboard is enabled, pygame rendering must be true
        self.rendering = params['rendering']  # true / false
        self.enable_keyboard = params['enable_keyboard']
        # --------------------------------------------------------------------------------------------------------------

        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.height = params['height']  # pygame camera window's height
        self.width = params['width']  # pygame camera window's width
        self.dt = params['dt']
        self.obs_range = params['obs_range']
        self.lidar_bin = params['lidar_bin']
        self.d_behind = params['d_behind']
        self.max_time_episode = params['max_time_episode']
        self.lidar_obs_size = int(self.obs_range / self.lidar_bin)
        self.max_ego_spawn_times = params['max_ego_spawn_times']

        # action and observation spaces
        """"""''''''''''''''''''''''''
        min_diff_point_ego_x = np.float32(- 0.5 )  # the minimum difference between x of a waypoint and x location of ego vehicle
        min_diff_point_ego_y = np.float32(- 0.5 )  # the minimum difference between y of a waypoint and y location of ego vehicle
        max_diff_point_ego_x = np.float32(  0.5 )  # the maximum difference between x of a waypoint and x location of ego vehicle
        max_diff_point_ego_y = np.float32(  0.5 )  # the maximum difference between y of a waypoint and y location of ego vehicle

        """"""''''''''''''''''''''''''

        self.action_space = spaces.Box(np.array([min_diff_point_ego_x,min_diff_point_ego_y]),
                                       np.array([max_diff_point_ego_x,max_diff_point_ego_y]),
                                       dtype=np.float32)

        """"""''''''''''''''''''''''''
        min_coord, max_coord = -1000, 1000  # in carla coordinates units
        min_sin_angle, max_sin_angle = -1, 1
        min_cos_angle, max_cos_angle = -1, 1
        min_speed, max_speed = -5, 10  # m/s speed, 10 is equivalent to 10*3.6 = 36 km/h
        """"""''''''''''''''''''''''''

        self.enable_lidar_observation = params['enable_lidar_observation']
        if self.enable_lidar_observation:
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=1, shape=(self.height, self.width, 3), dtype=np.uint8),
                'lidar': spaces.Box(low=0, high=1, shape=(self.lidar_obs_size, self.lidar_obs_size, 3), dtype=np.uint8),
                'state': spaces.Box(np.array([min_coord, min_coord, min_sin_angle, min_cos_angle, min_speed]),
                                    np.array([max_coord, max_coord, max_sin_angle, max_cos_angle, max_speed]),
                                    dtype=np.float32)
            }
        else:
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=1, shape=(self.height, self.width, 3), dtype=np.uint8),
                'state': spaces.Box(np.array([min_coord, min_coord, min_sin_angle, min_cos_angle, min_speed]),
                                    np.array([max_coord, max_coord, max_sin_angle, max_cos_angle, max_speed]),
                                    dtype=np.float32)
            }

        # self.observation_space = spaces.Dict(observation_space_dict)
        self.observation_space = observation_space_dict['camera']

        # Connect to carla server and get world object
        print('connecting to Carla server...')
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.world = client.load_world(params['town'])
        self.map = self.world.get_map()
        print('Carla server connected!')

        # Global Planner Config, random ego spawn should be [false] ----------------------------------------------------
        self.random_ego_spawn = params['random_ego_spawn']

        self.origin = params['origin']
        self.origin_transform = carla.Transform(carla.Location(x=self.origin['x'],
                                                               y=self.origin['y'],
                                                               z=self.origin['z']),
                                                carla.Rotation(pitch=self.origin['pitch'],
                                                               yaw=self.origin['yaw'],
                                                               roll=self.origin['roll']))

        self.destination = params['destination']
        self.destination_transform = carla.Transform(carla.Location(x=self.destination['x'],
                                                                    y=self.destination['y'],
                                                                    z=self.destination['z']),
                                                     carla.Rotation(pitch=self.destination['pitch'],
                                                                    yaw=self.destination['yaw'],
                                                                    roll=self.destination['roll']))

        # --------------------------------------------------------------------------------------------------------------

        self.keyboard_controller = None

        # spectator
        self.spectator_pose = params['spectator_pose']
        self.spectator = self.world.get_spectator()

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # --------------------------------------------------------------------------------------------------------------
        # Get spawn points
        self.vehicle_spawn_points = None
        self._add_vehicles_at_some_preferred_location()
        # --------------------------------------------------------------------------------------------------------------

        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_blueprint(params['ego_vehicle_filter'], color='49,8,8')

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Camera sensor
        self.camera_img = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.width))
        self.camera_bp.set_attribute('image_size_y', str(self.height))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # Lidar sensor
        if self.enable_lidar_observation:
            self.lidar_data = None
            self.lidar_height = 2.1
            self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
            self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            self.lidar_bp.set_attribute('channels', '32')
            self.lidar_bp.set_attribute('range', '5000')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # --------------------------------------------------------------------------------------------------------------
        # Hardcoded for now to save some GPU memory, however, pygame window works fine, even if rendering is disabled
        # if self.rendering:
        #     self.settings.no_rendering_mode = False
        # else:
        self.settings.no_rendering_mode = True
        # --------------------------------------------------------------------------------------------------------------

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
        self.time_step = 0
        self.dx = 0
        self.dy = 0
        self.prev_x = 0
        self.prev_y = 0

        self.helper = None
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.vehicle_front = None

        # Controllers --------------------------------------------------------------------------------------------------
        MPS_TO_KMH = 3.6
        self.target_speed = params['target_speed'] * MPS_TO_KMH
        self.ego_wheelbase = params['wheelbase']
        self.controller_type = params['controller_type'].replace(' ', '').lower()
        self.pure_pursuit_controller = None
        self.stanley_controller = None
        self.linear_mpc_controller = None
        # --------------------------------------------------------------------------------------------------------------

        # Initialize the renderer
        self._init_renderer()

    def reset(self):
        # Clear sensor objects
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*',
                                'controller.ai.walker', 'walker.*'])

        # Disable sync mode
        self._set_synchronous_mode(False)

        # Spawn surrounding vehicles
        self._add_vehicles_at_some_preferred_location()
        random.shuffle(self.vehicle_spawn_points)
        count = random.choice(range(self.number_of_vehicles, self.number_of_vehicles+1))
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                count -= 1

        if self.random_ego_spawn:
            # Spawn the ego vehicle
            ego_spawn_times = 0
            while True:
                if ego_spawn_times > self.max_ego_spawn_times:
                    self.reset()

                transform = random.choice(self.vehicle_spawn_points)

                if self._try_spawn_ego_vehicle_at(transform):
                    break
                else:
                    ego_spawn_times += 1
                    time.sleep(0.1)
        else:
            transform = carla.Transform(carla.Location(x=self.origin_transform.location.x,
                                                       y=self.origin_transform.location.y,
                                                       z=self.origin_transform.location.z),
                                       carla.Rotation(pitch=self.origin_transform.rotation.pitch,
                                                      yaw=self.origin_transform.rotation.yaw,
                                                      roll=self.origin_transform.rotation.roll))
            if not self._try_spawn_ego_vehicle_at(transform):
                self.reset()

        self.helper = Helper(self.rendering,
                             self.ego,
                             self.height,
                             self.width,
                             self.lidar_obs_size,
                             self.target_speed,
                             self.origin_transform,
                             self.destination_transform)  # singleton

        # Initialize controllers ---------------------------------------------------------------------------------------
        self.pure_pursuit_controller = PurePursuitController(self.helper.global_waypoints_xyv_np, self.dt, self.ego_wheelbase, self.target_speed)
        self.stanley_controller = StanleyController(self.helper.global_waypoints_xyv_np, self.dt, self.target_speed)
        self.linear_mpc_controller = LinearMPCController(self.helper.global_waypoints_xyv_np, self.dt, self.ego_wheelbase, self.target_speed)
        # --------------------------------------------------------------------------------------------------------------

        # update spectator
        self._update_spectator()

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        # Add lidar sensor
        if self.enable_lidar_observation:
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

            def get_lidar_data(data):
                self.lidar_data = data

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        if self.enable_keyboard:
            self.keyboard_controller = KeyboardControl(self.ego, start_in_autopilot=False)

        return self._get_obs()

    def record_step(self, clock):
        if self.enable_keyboard:
            pygame_quit = self.keyboard_controller.parse_events(self.world, self.ego, clock)
        else:
            pygame_quit = False

        if self.time_step == 0:
            self.prev_x, self.prev_y, _ = self.helper.get_ego_pos()
            self.dx, self.dy = 0, 0
        else:
            current_x, current_y, _ = self.helper.get_ego_pos()
            self.dx = current_x - self.prev_x
            self.dy = current_y - self.prev_y
            self.prev_x, self.prev_y = current_x, current_y

        record_done = self.helper.save_record(self.camera_img,
                                              self.dx,
                                              self.dy,
                                              self.ego.get_control().steer,
                                              self.ego.get_control().throttle,
                                              self.time_step)

        self.world.tick()  # updates sensors feed

        self._display_camera(self.camera_img)

        # Display on pygame
        self._pygame_display_flip()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        if (self.time_step % 10) == 0:
            self._update_spectator_no_delay()

        return pygame_quit, None, (record_done or self._terminal()), {}

    def step(self, action, method):
        """
        1. no model dxdy
        2. learning dxdy
        3. no model and learning throttle and steer
        """

        if method == 1:
            self._apply_control_step_dxdy_no_model(action)
        elif method == 2:
            self._apply_control_step_dxdy_learning(action)
        elif method == 3:
            self._apply_control_step_throttle_steer(action)

        self.world.tick()  # updates sensors feed

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        return self._get_obs(), self._get_reward(), self._terminal(), {}

    def _apply_control_step_dxdy_no_model(self, action):
        # default values
        throttle, steer, brake = 0, 0, 0

        current_x, current_y, current_yaw = self.helper.get_ego_pos()
        current_speed = self.helper.get_ego_speed()

        # --------------------------------------------------------------------------------------------------------------
        # # A. use it with no model
        new_waypoints = self.helper.get_new_waypoints(current_x, current_y)
        # --------------------------------------------------------------------------------------------------------------

        if self.controller_type == 'pure_pursuit':
            self.pure_pursuit_controller.update_waypoints(new_waypoints)
            self.pure_pursuit_controller.update_values(current_x, current_y, current_yaw, current_speed, self.time_step)
            self.pure_pursuit_controller.update_controls()
            throttle, steer, brake = self.pure_pursuit_controller.get_commands()
        elif self.controller_type == 'stanley':
            self.stanley_controller.update_waypoints(new_waypoints)
            self.stanley_controller.update_values(current_x, current_y, current_yaw, current_speed, self.time_step)
            self.stanley_controller.update_controls()
            throttle, steer, brake = self.stanley_controller.get_commands()
        elif self.controller_type == 'linear_mpc':
            self.linear_mpc_controller.update_waypoints(new_waypoints)
            self.linear_mpc_controller.update_values(current_x, current_y, current_yaw, current_speed, self.time_step)
            self.linear_mpc_controller.update_controls()
            throttle, steer, brake = self.linear_mpc_controller.get_commands()

        self.ego.apply_control(carla.VehicleControl(throttle, steer, brake))

        if (self.time_step % 50) == 0:
            print(f'total timesteps {self.total_step:.0f}, '  # total timesteps spent across episodes
                  f'resetstep {self.reset_step:.0f}, '
                  f'timestep {self.time_step:.0f}, '
                  f'dx {action[0]:.3f}, '
                  f'dy {action[1]:.3f}, '
                  # -ve values: steer to the left
                  f'steer {steer:.3f}, '
                  # +ve values: steer to the right
                  f'throttle {throttle:.3f}, '
                  f'brake {brake:.3f}')

    def _apply_control_step_dxdy_learning(self, action):
        # default values
        throttle, steer, brake = 0, 0, 0

        current_x, current_y, current_yaw = self.helper.get_ego_pos()
        current_speed = self.helper.get_ego_speed()

        # --------------------------------------------------------------------------------------------------------------
        # B. use it with learning models
        new_waypoints = self.helper.get_new_waypoints_LR(current_x, current_y, action)
        # --------------------------------------------------------------------------------------------------------------

        if self.controller_type == 'pure_pursuit':
            self.pure_pursuit_controller.update_waypoints(new_waypoints)
            self.pure_pursuit_controller.update_values(current_x, current_y, current_yaw, current_speed, self.time_step)
            self.pure_pursuit_controller.update_controls()
            throttle, steer, brake = self.pure_pursuit_controller.get_commands()
        elif self.controller_type == 'stanley':
            self.stanley_controller.update_waypoints(new_waypoints)
            self.stanley_controller.update_values(current_x, current_y, current_yaw, current_speed, self.time_step)
            self.stanley_controller.update_controls()
            throttle, steer, brake = self.stanley_controller.get_commands()
        elif self.controller_type == 'linear_mpc':
            self.linear_mpc_controller.update_waypoints(new_waypoints)
            self.linear_mpc_controller.update_values(current_x, current_y, current_yaw, current_speed, self.time_step)
            self.linear_mpc_controller.update_controls()
            throttle, steer, brake = self.linear_mpc_controller.get_commands()

        self.ego.apply_control(carla.VehicleControl(throttle, steer, brake))

        if (self.time_step % 50) == 0:
            print(f'total timesteps {self.total_step:.0f}, '  # total timesteps spent across episodes
                  f'resetstep {self.reset_step:.0f}, '
                  f'timestep {self.time_step:.0f}, '
                  f'dx {action[0]:.3f}, '
                  f'dy {action[1]:.3f}, '
                  # -ve values: steer to the left
                  f'steer {steer:.3f}, '
                  # +ve values: steer to the right
                  f'throttle {throttle:.3f}, '
                  f'brake {brake:.3f}')

    def _apply_control_step_throttle_steer(self, steer=0, throttle=0.2):
        # default values
        throttle, steer, brake = throttle, steer, 0

        self.ego.apply_control(carla.VehicleControl(throttle, steer, brake))

        if (self.time_step % 50) == 0:
            print(f'total timesteps {self.total_step:.0f}, '  # total timesteps spent across episodes
                  f'resetstep {self.reset_step:.0f}, '
                  f'timestep {self.time_step:.0f}, '
                  # -ve values: steer to the left
                  f'steer {steer:.3f}, '
                  # +ve values: steer to the right
                  f'throttle {throttle:.3f}, '
                  f'brake {brake:.3f}')

    def _get_obs(self):
        ## Display camera image
        self._display_camera(self.camera_img)

        if self.enable_lidar_observation:
            ## Lidar image generation
            point_cloud = []
            # Get point cloud data
            for location in self.lidar_data:
                point_cloud.append([location.point.x, location.point.y, -location.point.z])
            point_cloud = np.array(point_cloud)
            # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
            # and z is set to be two bins.
            y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
            x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
            z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
            # Get lidar image according to the bins
            lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
            lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
            lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
            # Add the waypoints to lidar image
            wayptimg = np.zeros((self.lidar_obs_size, self.lidar_obs_size), dtype=np.uint8)  # Equal to a zero matrix
            wayptimg = np.expand_dims(wayptimg, axis=2)
            wayptimg = np.fliplr(np.rot90(wayptimg, 3))

            # Get the final lidar image
            lidar = np.concatenate((lidar, wayptimg), axis=2)
            lidar = np.flip(lidar, axis=0)
            # lidar = np.rot90(lidar, 1)

            # Display lidar image
            self._display_lidar(lidar * 255)

            obs = {
                'camera': self.camera_img.astype(np.uint8),
                'lidar': lidar,
                # 'state': self._get_state(),
            }

        else:
            obs = {
                'camera': self.camera_img.astype(np.uint8),
                # 'state': self._get_state(),
            }

        # Display on pygame
        self._pygame_display_flip()

        img = Helper.img_preprocess(obs['camera']).reshape(1, -1)

        return img

    def _get_reward(self):
        r_collision     = 0
        r_speed         = 0
        r_fast          = 0
        r_out           = 0
        r_destination   = 0
        r_steer         = 0
        r_lat           = 0
        r_other_dir     = 0

        closest_index, closest_distance = self.helper.find_closest_point_index_on_global_route()
        v = self.ego.get_velocity()
        total_speed = np.sqrt(v.x ** 2 + v.y ** 2)

        # get longitudinal speed component
        longitudinal_velocity_component = np.dot(self.helper.get_velocity_vector(v),
                                                 self.helper.get_closest_yaw_vector(closest_index))

        """Calculate the step reward."""

        # reward if moving in the other direction
        if self.helper.is_moving_in_other_direction():
            pass
            # r_other_dir = -1

        # reward for speed tracking
        # r_speed = -abs(total_speed - self.target_speed)

        # reward for collision
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for steering "squared":
        # r_steer = -self.ego.get_control().steer ** 2

        # reward for being very far away from the route of the global path planner
        if self.helper.is_away_from_global_route(closest_distance):
            pass
            # r_out = -1

        # cost for too fast
        if longitudinal_velocity_component > self.target_speed:
            pass
            # r_fast = -1

        # cost for lateral acceleration (r * V^2)
        # r_lat = - abs(self.ego.get_control().steer) * longitudinal_velocity_component ** 2

        # cost for distance from destination
        r_destination = - self.helper.distance_to_destination()

        r = 20.0            *               r_collision          +       \
            5.0             *               r_other_dir          +       \
            1.0             *               r_fast               +       \
            1.0             *               r_speed              +       \
            1.0             *               r_steer              +       \
            1.0             *               r_out                +       \
            1.0             *               r_destination        +       \
            0.2             *               r_lat

        return r

    def _terminal(self):
        """Calculate whether to terminate the current episode."""

        # If vehicle became out of the road boundaries
        if self.helper.is_out_of_road_boundaries():
            return True

        # If moving in the other direction
        # if self.helper.is_moving_in_other_direction():
        #     return True

        # If collides
        if len(self.collision_hist) > 0:
            return True

        # If reach maximum timestep per episode
        if self.time_step > self.max_time_episode:
            return True

        # If at destination
        if self.helper.is_at_destination():
            self.ego.apply_control(carla.VehicleControl(0, 0, 1))  # throttle, steer, brake
            return True

        # # If very far away from the route of the global path planner
        # closest_index, closest_distance = self.helper.find_closest_point_index_on_global_route()
        # if self.helper.is_away_from_global_route(closest_distance):
        #     return True

        return False

    def _create_vehicle_blueprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """
        Create the blueprint for a specific actor type.
        :param actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
        :return bp: the blueprint object of carla.
        """

        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = blueprint_library[0]  # random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _try_spawn_ego_vehicle_at(self, transform):
        """
        Try to spawn the ego vehicle at specific transform.
        :param transform: the carla transform object.
        :return Bool indicating whether the spawn is successful.
        """

        vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            time.sleep(1)
            self.ego = vehicle

            if self.enable_autopilot:
                self.ego.set_autopilot()
                self.ego.enable_constant_velocity(carla.Vector3D(0.5, 0.0, 0.0))
            time.sleep(1)  # if commented spawned vehicles might not move at all
            return True

        return False

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """
        Try to spawn a surrounding vehicle at specific transform with random blueprint.
        :param transform: the carla transform object.
        :return Bool indicating whether the spawn is successful.
        ref: https://carla.readthedocs.io/en/0.9.6/python_api_tutorial/
        """

        blueprint = self._create_vehicle_blueprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            if self.let_other_vehicles_move:
                time.sleep(1)
                vehicle.set_autopilot()
                vehicle.enable_constant_velocity(carla.Vector3D(1.5, 0.0, 0.0))
                time.sleep(1)  # if commented spawned vehicles might not move at all
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """
        Try to spawn a walker at specific transform with random blueprint.
        :param transform: the carla transform object.
        :return Bool indicating whether the spawn is successful.
        """

        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('stanley_controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _update_spectator(self):
        # Let's position the spectator just behind the vehicle
        # Carla.Transform has two parameters - Location and Rotation. We use this to
        # position the spectator by going 5 metres behind and 2.5 metres above the
        # ego_vehicle

        # static spectator - doesn't update as vehicle move

        time.sleep(1)

        transform = carla.Transform(self.ego.get_transform().transform(carla.Location(x=+self.spectator_pose['add_x'],
                                                                                      y=+self.spectator_pose['add_y'],
                                                                                      z=self.spectator_pose['z'])),
                                                                       carla.Rotation(pitch=self.spectator_pose['pitch'],
                                                                                      yaw=self.helper.get_ego_pos()[2] * 57.29577951308232,
                                                                                      roll=self.spectator_pose['roll']))
        self.spectator.set_transform(transform)

    def _update_spectator_no_delay(self):
        # 180/np.pi = 57.29577951308232
        # dynamic spectator - updates as vehicle moves

        transform = carla.Transform(self.ego.get_transform().transform(carla.Location(x=-10,
                                                                                      y=+0,
                                                                                      z=+50)),
                                                                       carla.Rotation(pitch=-60,
                                                                                      yaw=self.helper.get_ego_pos()[2] * 57.29577951308232,
                                                                                      roll=0))

        self.spectator.set_transform(transform)

    def _init_renderer(self):
        """
        Initialize the render

        width is multiplied by 2, because we have two renders
        - one for camera
        - and one for lidar
        """
        if self.rendering:
            pygame.init()
            if self.enable_lidar_observation:
                self.display = pygame.display.set_mode(
                  (self.width * 2 * PYGAME_SCALE_FACTOR, self.height * PYGAME_SCALE_FACTOR),
                  pygame.HWSURFACE | pygame.DOUBLEBUF)
            else:
                self.display = pygame.display.set_mode(
                    (self.width * PYGAME_SCALE_FACTOR, self.height * PYGAME_SCALE_FACTOR),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)

    def _set_synchronous_mode(self, synchronous=True):
        """
        Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()

    def _display_camera(self, camera):
        """
        Note that:
            camera is displayed on the left
            lidar is displayed on the right
            that's fine, but if you would like to display camera on the right
            you have to replace the line below:
            original: self.display.blit(camera_surface, (0, 0))
            modified: self.display.blit(camera_surface, (self.width, 0))
        """
        if self.rendering:
            camera_surface = self.helper.rgb_to_display_surface(camera)
            self.display.blit(camera_surface, (0, 0))

    def _display_lidar(self, lidar):
        """
        Note that:
            camera is displayed on the left
            lidar is displayed on the right
            that's fine, but if you would like to display lidar on the left
            you have to replace the line below:
            original: self.display.blit(lidar_surface, (self.width, 0))
            modified: self.display.blit(lidar_surface, (0, 0))
        """
        if self.rendering:
            lidar_surface = self.helper.lidar_to_display_surface(lidar)
            # lidar_surface = Helper.rgb_to_display_surface(lidar, self.width, self.height)
            self.display.blit(lidar_surface, (self.width, 0))

    def _pygame_display_flip(self):
        if self.rendering:
            pygame.display.flip()

    def _add_vehicles_at_some_preferred_location(self):
        # --------------------------------------------------------------------------------------------------------------
        self.vehicle_spawn_points = list(self.map.get_spawn_points())
        # --------------------------------------------------------------------------------------------------------------
        """
        overwrite the original self.vehicle_spawn_points
        """
        # --------------------------------------------------------------------------------------------------------------
        y1 = random.uniform(-95.0, -90.0)
        y2 = y1 + random.uniform(30, 35)
        self.vehicle_spawn_points = [
                                        carla.Transform(carla.Location(x=-84.5,
                                                                       y=y1,
                                                                       z=0.3),
                                                        carla.Rotation(pitch=0.000000,
                                                                       yaw=random.uniform(80.0, 100.0),
                                                                       roll=0.000000)),


                                        carla.Transform(carla.Location(x=-87.5,
                                                                       y=y2,
                                                                       z=0.3),
                                                        carla.Rotation(pitch=0.000000,
                                                                       yaw=random.uniform(80.0, 100.0),
                                                                       roll=0.000000)),
            ]
        # --------------------------------------------------------------------------------------------------------------
