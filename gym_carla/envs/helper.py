import cv2
import copy
import datetime
import numpy as np
import pandas as pd
import math
import pygame
import carla
# GlobalRoutePlannerDAO is a Data Access Object interface for GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner
from config.config import *

GRP_SAMPLING_RESOLUTION = 1  # sampling resolution of points generated by the global route planner
OUT_OF_WAY_THRESHOLD = 7.5
# Path interpolation parameters
# lookahead path
DIST_THRESHOLD_TO_LAST_WAYPOINT = 10.0  # some distance from last position before simulation ends
INTERP_LOOKAHEAD_DISTANCE = 20  # lookahead in meters
INTERP_DISTANCE_RES = 0.01  # distance between interpolated points

PYGAME_SCALE_FACTOR = 3  # scale factor for pygame window, doesn't affect the original image resolution

class Helper:
    __singleton = None
    __initialized = None

    def __new__(cls, *args, **kwargs):
        if not cls.__singleton:
            cls.__singleton = super(Helper, cls).__new__(cls)
        return cls.__singleton

    def __init__(self, rendering,
                 ego_vehicle, height, width, lidar_obs_size,
                 target_speed,
                 origin_transform,
                 destination_transform):

        self._ego_vehicle = ego_vehicle
        self._ego_vehicle_id = self._ego_vehicle.id

        if not self.__initialized:
            self.__initialized = True
            super().__init__()
            self._world = self._ego_vehicle.get_world()
            self._map = self._world.get_map()
            self.height = height
            self.width = width

            self.record_data_reset = {'img': [],
                                      'dx':  [],
                                      'dy':  [],
                                      'steer': [],
                                      'throttle': []}
            self.record_data = copy.deepcopy(self.record_data_reset)

            self.record_csv_path = record_saving_path + '/log.csv'
            if not os.path.exists(self.record_csv_path):
                record_data_pd = pd.DataFrame(self.record_data)
                record_data_pd.to_csv(self.record_csv_path, index=False)  # putting headers in the csv file first

            self.lidar_obs_size = lidar_obs_size
            self.rendering = rendering  # true / false
            if self.rendering:
                self.camera_surface = pygame.Surface((self.width, self.height)).convert()
                self.lidar_surface = pygame.Surface((self.lidar_obs_size, self.lidar_obs_size)).convert()

            self._target_speed = target_speed
            self._origin_transform = origin_transform
            self._destination_transform = destination_transform
            self._grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(wmap=self._world.get_map(), sampling_resolution=GRP_SAMPLING_RESOLUTION))
            self.global_waypoints = self._grp.trace_route(self._origin_transform.location, self._destination_transform.location)

            # sometimes there is duplication in some points
            self._solve_waypoints_duplication_issue()
            # x, y, yaw
            self.global_waypoints_xyh_np = self.waypoints_data_structure_conversion_xyh(self.global_waypoints)
            # sometimes interpolation is needed
            self._solve_waypoints_inter_distances_issue()
            # x, y, v
            self.global_waypoints_xyv_np = self.global_waypoints_xyh_np.copy()
            self.global_waypoints_xyv_np[:, 2] = self._target_speed

            assert self.global_waypoints_xyh_np.shape[0] == self.global_waypoints_xyv_np.shape[0]

            # self._plot_waypoints()
            # self._plot_waypoints_np()

            # print the maximum steering angle of each wheel of the vehicle, and update them in the controllers used
            # self.get_ego_max_steering_angle()  # [deg] retrieved from carla simulator for vehicle audi a2

            # MODULE 7 -------------------------------------------------------------------------------------------------
            self.wp_distance = []   # distance array
            # Linearly interpolate between waypoints and store in a list
            # interpolated values rows = waypoints, columns = [x, y, v])
            self.wp_interp = np.empty((0, self.global_waypoints_xyv_np.shape[1]), dtype=float)
            # hash table which indexes waypoints_np to the index of the waypoint in wp_interp
            self.wp_interp_hash = np.empty(0, dtype=int)
            self.module_7()
            # ----------------------------------------------------------------------------------------------------------

            # register origin and destination waypoints
            self.origin_waypoint = None
            self.destination_waypoint = None
            self._register_origin_destination()

    def module_7(self):
        # Because the waypoints are discrete and our controller performs better
        # with a continuous path, here we will send a subset of the waypoints
        # within some lookahead distance from the closest point to the vehicle.
        # Interpolating between each waypoint will provide a finer resolution
        # path and make it more "continuous". A simple linear interpolation
        # is used as a preliminary method to address this issue, though it is
        # better addressed with better interpolation methods (spline
        # interpolation, for example).
        # More appropriate interpolation methods will not be used here for the
        # sake of demonstration on what effects discrete paths can have on
        # the controller. It is made much more obvious with linear
        # interpolation, because in a way part of the path will be continuous
        # while the discontinuous parts (which happens at the waypoints) will
        # show just what sort of effects these points have on the controller.
        # Can you spot these during the simulation? If so, how can you further
        # reduce these effects?

        # Linear interpolation computations

        # Compute differences along each axis separately col1[idx2] - col1[idx1], and col2[idx2] - col2[idx1], etc ...
        diff = np.diff(self.global_waypoints_xyv_np[:, :2], axis=0)

        # Compute distances using the Euclidean norm
        self.wp_distance = np.sqrt(np.sum(diff ** 2, axis=1))

        # Append 0 to the end of the distance array
        # last distance is 0 because it is the distance from the last waypoint to the last waypoint
        self.wp_distance = np.append(self.wp_distance, 0)

        interp_counter = 0  # counter for current interpolated point index

        for i in range(self.global_waypoints_xyv_np.shape[0] - 1):
            # Add original waypoint to interpolated waypoints list (and append it to the hash table)
            self.wp_interp = np.vstack([self.wp_interp, self.global_waypoints_xyv_np[i]])
            self.wp_interp_hash = np.append(self.wp_interp_hash, interp_counter)
            interp_counter += 1

            # Interpolate to the next waypoint
            wp_vector = self.global_waypoints_xyv_np[i + 1] - self.global_waypoints_xyv_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector)

            # Compute interpolated points
            num_pts_to_interp = int(np.floor(self.wp_distance[i] / float(INTERP_DISTANCE_RES)) - 1)
            next_wp_vectors = np.arange(1, num_pts_to_interp + 1)[:, np.newaxis] * INTERP_DISTANCE_RES * wp_uvector
            interpolated_points = self.global_waypoints_xyv_np[i] + next_wp_vectors

            # Append interpolated points to wp_interp and update the counter
            self.wp_interp = np.vstack([self.wp_interp, interpolated_points])
            interp_counter += num_pts_to_interp

        # Add the last waypoint at the end
        self.wp_interp = np.vstack([self.wp_interp, self.global_waypoints_xyv_np[-1]])
        self.wp_interp_hash = np.append(self.wp_interp_hash, interp_counter)

    def get_new_waypoints(self, current_x, current_y):
        # To reduce the amount of waypoints sent to the controller,
        # provide a subset of waypoints that are within some
        # lookahead distance from the closest point to the car. Provide
        # a set of waypoints behind the car as well.

        # Find the closest waypoint index to car. First increment the index
        # from the previous index until the new distance calculations
        # are increasing. Apply the same rule decrementing the index.
        # The final index should be the closest point (it is assumed that
        # the car will always break out of instability points where there
        # are two indices with the same minimum distance, as in the center of a circle)

        car_position = np.array([current_x, current_y])

        # Vectorized distance calculations
        distances = np.linalg.norm(self.global_waypoints_xyv_np[:, :2] - car_position, axis=1)

        # Find the closest waypoint index
        closest_index = np.argmin(distances)

        # Once the closest index is found, return the path that has 1
        # waypoint behind and X waypoints ahead, where X is the index
        # that has a lookahead distance specified by
        # INTERP_LOOKAHEAD_DISTANCE

        # Determine indices within lookahead distance ahead
        lookahead_indices = np.where(np.cumsum(self.wp_distance[closest_index:]) >= INTERP_LOOKAHEAD_DISTANCE)[0]

        if lookahead_indices.shape[0] > 0:
            lookahead_index = lookahead_indices[0]
        else:  # the sum of the remaining points in the array won't meet the condition,
               # so we take the last index instead
            lookahead_index = self.global_waypoints_xyv_np.shape[0] - 1 - closest_index

        # Find subset indices
        waypoint_subset_first_index = max(closest_index - 1, 0)  # closest_index -1 instead of just closest_index
                                                                 # to take one point behind the vehicle,
                                                                 # stabilizes the controller
        waypoint_subset_last_index = closest_index + lookahead_index

        # Use the first and last waypoint subset indices into the hash
        # table to obtain the first and last indices for the interpolated
        # list. Update the interpolated waypoints to the controller
        # for the next controller update.
        # Use the subset indices to obtain the waypoints
        new_waypoints = self.wp_interp[self.wp_interp_hash[waypoint_subset_first_index]:
                                       self.wp_interp_hash[waypoint_subset_last_index] + 1]

        return new_waypoints

    def get_new_waypoints_LR(self, current_x, current_y, action):
        delta_x, delta_y = action

        current_speed = self.get_ego_speed()  # km/h

        # source
        car_position_speed = np.array([current_x, current_y, current_speed])

        # destination
        waypoint_position_speed = np.array([current_x + delta_x, current_y + delta_y, self._target_speed])

        distance = delta_x ** 2 + delta_y ** 2
        num_pts_to_interp = int((distance / float(INTERP_DISTANCE_RES)) - 1)
        num_pts_to_interp = max(num_pts_to_interp, max(num_pts_to_interp, 1))

        wp_vector = waypoint_position_speed[:2] - car_position_speed[:2]
        norm_wp_vector = np.linalg.norm(wp_vector)

        if norm_wp_vector == 0:
            next_wp_vectors = np.empty((0, 2))
        else:
            wp_uvector = wp_vector / norm_wp_vector
            next_wp_vectors = np.arange(1, num_pts_to_interp + 1)[:, np.newaxis] * INTERP_DISTANCE_RES * wp_uvector

        if norm_wp_vector == 0:
            next_speed_vectors = np.empty((0, 1))  # empty of shape (0, 1)
        else:
            interp_speed_res = (current_speed - self._target_speed) / num_pts_to_interp
            speed_vector = waypoint_position_speed[2] - car_position_speed[2]
            speed_uvector = speed_vector / np.linalg.norm(speed_vector)
            next_speed_vectors = np.arange(1, num_pts_to_interp + 1)[:, np.newaxis] * interp_speed_res * speed_uvector

        next_vectors = np.hstack((next_wp_vectors, next_speed_vectors))

        interpolated_points = car_position_speed + next_vectors

        # Vectorized distance calculations
        distances = np.linalg.norm(self.global_waypoints_xyv_np[:, :2] - car_position_speed[:2], axis=1)

        # Find the closest waypoint index
        closest_index = max( np.argmin(distances) - 1 , 0 )  # -1 to take a point behind the car, stabilizes controller
        point_behind_ego = self.wp_interp[self.wp_interp_hash[closest_index]]

        point_behind_ego[2] = current_speed  # modify speed to correct value

        final_waypoints = np.empty((0, self.global_waypoints_xyv_np.shape[1]), dtype=float)
        final_waypoints = np.vstack([final_waypoints, point_behind_ego, interpolated_points])

        return final_waypoints

    def _register_origin_destination(self):
        temp_origin = carla.Location(x=self.global_waypoints_xyh_np[0, 0], y=self.global_waypoints_xyh_np[0, 1], z=0.06)
        temp_destination = carla.Location(x=self.global_waypoints_xyh_np[-1, 0], y=self.global_waypoints_xyh_np[-1, 1], z=0.06)
        self.origin_waypoint = self._map.get_waypoint(temp_origin)
        print('origin', self.origin_waypoint.transform.location)
        self.destination_waypoint = self._map.get_waypoint(temp_destination)
        print('destination', self.destination_waypoint.transform.location)

    def _solve_waypoints_duplication_issue(self):
        i = 0
        LEN = len(self.global_waypoints)
        while i < LEN - 1:
            pos_0 = self.get_pos(self.global_waypoints[i][0])
            pos_1 = self.get_pos(self.global_waypoints[i + 1][0])
            if pos_0 == pos_1:
                self.global_waypoints.pop(i)
                LEN -= 1
            else:
                i += 1

    def _solve_waypoints_inter_distances_issue(self):
        i = 0
        LEN = self.global_waypoints_xyh_np.shape[0]
        while i < LEN - 1:
            pos_0 = self.global_waypoints_xyh_np[i]
            pos_1 = self.global_waypoints_xyh_np[i + 1]
            norm = np.linalg.norm([pos_0[0] - pos_1[0], pos_0[1] - pos_1[1]])
            if norm > 1.51 * GRP_SAMPLING_RESOLUTION:
                interpolated_pos = np.zeros((1, 3))
                interpolated_pos[0][0] = (pos_0[0]+pos_1[0])/2   # x
                interpolated_pos[0][1] = (pos_0[1]+pos_1[1])/2   # y
                interpolated_pos[0][2] = (pos_0[2]+pos_1[2])/2   # yaw in rads
                self.global_waypoints_xyh_np = np.insert(self.global_waypoints_xyh_np, i+1, interpolated_pos, axis=0)
                LEN += 1
            else:
                i += 1

    def _plot_waypoints(self):
        T = -1  # Time before line disappears, negative for never
        for pt in self.global_waypoints:
            temp = carla.Location(x=pt[0].transform.location.x, y=pt[0].transform.location.y, z=0.5)
            # self.world.debug.draw_line(ego_pos, temp, thickness=0.2, life_time=T, color=carla.Color(r=255, g=0, b=0))
            self._world.debug.draw_point(temp, color=carla.Color(r=255, g=0, b=0), life_time=T)

    def _plot_waypoints_np(self):
        T = -1  # Time before line disappears, negative for never
        for pt in self.global_waypoints_xyh_np:
            temp = carla.Location(x=pt[0], y=pt[1], z=0.5)
            # self.world.debug.draw_line(ego_pos, temp, thickness=0.2, life_time=T, color=carla.Color(r=255, g=0, b=0))
            self._world.debug.draw_point(temp, color=carla.Color(r=255, g=0, b=0), life_time=T)

    def distance_to_destination(self):
        ego_x, ego_y, _ = self.get_ego_pos()
        x, y, _ = self.get_pos(self.destination_waypoint)
        return np.linalg.norm([ego_x - x, ego_y - y])

    def is_at_destination(self):
        return self.distance_to_destination() < DIST_THRESHOLD_TO_LAST_WAYPOINT

    def get_ego_max_steering_angle(self):
        """
        in degrees
        """
        for wheel in self._ego_vehicle.get_physics_control().wheels:
            print(wheel.max_steer_angle)

    def get_ego_pos(self):
        trans = self._ego_vehicle.get_transform()
        x = trans.location.x
        y = trans.location.y
        yaw = trans.rotation.yaw / 180 * np.pi
        return x, y, yaw

    @staticmethod
    def get_pos(waypoint):
        trans = waypoint.transform
        x = trans.location.x
        y = trans.location.y
        yaw = trans.rotation.yaw / 180 * np.pi
        return x, y, yaw

    def is_out_of_road_boundaries(self):
        l1 = self._ego_vehicle.get_location()
        l2 = self._map.get_waypoint(l1).transform.location
        d = np.array([l1.x - l2.x, l1.y - l2.y, l1.z - l2.z])
        distance = np.linalg.norm(d)
        return distance > 5

    def get_ego_speed(self, kmh=True):
        """in km/h"""
        vector = self._ego_vehicle.get_velocity()
        if kmh:
            MPS_TO_KMH = 3.6
            return MPS_TO_KMH * math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)
        else:
            return math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)

    def find_closest_point_index_on_global_route(self):
        ego_x, ego_y, _ = self.get_ego_pos()
        car_position = np.array([ego_x, ego_y])

        # Vectorized distance calculations
        distances = np.linalg.norm(self.global_waypoints_xyv_np[:, :2] - car_position, axis=1)

        # Find the closest waypoint index
        closest_index = np.argmin(distances)

        return closest_index, distances[closest_index]

    def is_moving_in_other_direction(self):
        """
        Is the vehicle approaching a half circle maneuver
        i.e., going back to the origin instead of the destination?
        """
        _, _, yaw1 = self.get_ego_pos()  # in rads
        yaw1_vec = self.get_yaw_vector(yaw1)

        waypoint = self._map.get_waypoint(self._ego_vehicle.get_location())
        yaw2 = waypoint.transform.rotation.yaw  # in degs
        yaw2_vec = self.get_yaw_vector(yaw2  / 180 * np.pi)

        cos_angle = np.dot(yaw1_vec, yaw2_vec)
        # is angle between waypoint and vehicle yaw > 75 degrees?
        # most vehicles in carla have their max steering angle as 70 deg,
        # so it is better not to choose something smaller than 70,
        # not to limit steering capability of the vehicle
        return cos_angle < 0.25

    def is_away_from_global_route(self, distance):
        """
        Did the vehicle became very faraway from the path generated by the global path planner?
        """
        return distance > OUT_OF_WAY_THRESHOLD

    def get_closest_yaw_vector(self, closest_index):
        _, _, yaw = self.global_waypoints_xyh_np[closest_index]  # yaw in rads
        return self.get_yaw_vector(yaw)

    def get_velocity_vector(self, v):
        return np.array([v.x, v.y])

    def get_yaw_vector(self, yaw):
        """
        :param yaw: in rads
        :return:
        """
        yaw_vector = np.array([np.cos(yaw), np.sin(yaw)])
        return yaw_vector

    def closest_waypoint_to_ego(self):
        min_distance = float('inf')
        closest_index = -1
        for i, combo in enumerate(self.global_waypoints):
            waypoint, road_option = combo
            distance = waypoint.transform.location.distance(
                self._ego_vehicle.get_location())
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return self.global_waypoints[closest_index][0]

    def waypoints_of_vehicles(self):
        """
        :return: numpy 2D array of 3 columns for [x, y, yaw]
        """

        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")

        return_array = np.empty((len(vehicle_list), 3), dtype=float)

        for i in range(len(vehicle_list)):
            vehicle = vehicle_list[i]
            location = vehicle.get_location()
            rotation = vehicle.get_transform().rotation
            x = location.x
            y = location.y
            yaw = rotation.yaw

            return_array[i, :] = [x, y, yaw]

        return return_array

    def waypoints_data_structure_conversion_xyv(self, waypoints):
        """
        :param waypoints: list of waypoints
            each waypoint is a tuple of two elements
            first element: carla.libcarla.Waypoint object
            second element: RoadOption value
        :return: numpy 2D array of 3 columns for [x, y, target speed]
        """

        return_array = np.empty((len(waypoints), 3), dtype=float)

        for i in range(len(waypoints)):
            waypoint, _ = waypoints[i]
            x = waypoint.transform.location.x
            y = waypoint.transform.location.y

            return_array[i, :] = [x, y, self._target_speed]

        return return_array

    def rgb_to_display_surface(self, rgb):
        rgb = np.flip(rgb, axis=1)
        rgb = np.rot90(rgb, 1)
        pygame.surfarray.blit_array(self.camera_surface, rgb)
        if PYGAME_SCALE_FACTOR > 1:
            camera_surface = pygame.transform.scale(self.camera_surface, (self.width * PYGAME_SCALE_FACTOR, self.height * PYGAME_SCALE_FACTOR))
            return camera_surface
        else:
            return self.camera_surface

    def lidar_to_display_surface(self, lidar):
        display = np.flip(lidar, axis=1)
        display = np.rot90(display, 1)
        pygame.surfarray.blit_array(self.lidar_surface, display)
        return self.lidar_surface

    @staticmethod
    def waypoints_data_structure_conversion_xyh(waypoints):
        """
        :param waypoints: list of waypoints
            each waypoint is a tuple of two elements
            first element: carla.libcarla.Waypoint object
            second element: RoadOption value
        :return: numpy 2D array of 3 columns for [x, y, yaw]
        """

        return_array = np.empty((len(waypoints), 3), dtype=float)

        for i in range(len(waypoints)):
            waypoint, _ = waypoints[i]
            x = waypoint.transform.location.x
            y = waypoint.transform.location.y
            yaw = waypoint.transform.rotation.yaw  # in degs
            yaw = yaw / 180 * np.pi  # in rads

            return_array[i, :] = [x, y, yaw]

        return return_array

    def save_record(self, img, dx, dy, steer, throttle, total_step):
        filename = Helper.save_img_to_disk(img, total_step)
        self.record_data['img'].append(filename)
        self.record_data['dx'].append(dx)
        self.record_data['dy'].append(dy)
        self.record_data['steer'].append(steer)
        self.record_data['throttle'].append(throttle)

        if (total_step % 10) == 0:
            record_df = pd.DataFrame(self.record_data)
            record_df.to_csv(self.record_csv_path, mode='a', index=False, header=False)
            self.record_data = copy.deepcopy(self.record_data_reset)
            # print(total_step)
            if total_step >= 5_000:
                # sys.exit()
                return True
        return False

    @staticmethod
    def timestamp():
        return f"{datetime.datetime.now():%Y_%m_%d %H_%M_%S}"

    @staticmethod
    def save_img_to_disk(img, time_step):
        filename1 = f"{record_saving_path}/img/"
        filename2 = f"{Helper.timestamp()}_{time_step}.png"
        cv2.imwrite(filename1 + filename2, img)
        return filename2

    @staticmethod
    def img_preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # cv2.imshow('img1', img)
        # cv2.waitKey(0)
        img = img / 255
        # cv2.imshow('img2', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img

    # def is_any_vehicle_hazard(self):
    #     actor_list = self._world.get_actors()
    #     vehicle_list = actor_list.filter("*vehicle*")
    #
    #     ego_vehicle_location = self._ego_vehicle.get_location()
    #     ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
    #
    #     for target_vehicle in vehicle_list:
    #         # do not account for the ego vehicle
    #         if target_vehicle.id == self._ego_vehicle_id:
    #             continue
    #
    #         target_vehicle_location = target_vehicle.get_location()
    #
    #         # if the object is not in our lane it's not an obstacle
    #         target_vehicle_waypoint = self._map.get_waypoint(target_vehicle_location)
    #         if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
    #                 target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
    #             continue
    #
    #         if self.is_within_distance_ahead(target_vehicle_location, ego_vehicle_location,
    #                                          self._ego_vehicle.get_transform().rotation.yaw,
    #                                          PROXIMITY_THRESHOLD):
    #             return True
    #
    #     return False
    #
    # @staticmethod
    # def is_within_distance_ahead(target_location, current_location, current_orientation, proximity_threshold):
    #     target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    #     norm_target = np.linalg.norm(target_vector)
    #     if norm_target > proximity_threshold:
    #         return False
    #
    #     in_vicinity_of_target = norm_target < 1
    #     if in_vicinity_of_target:
    #         return True
    #
    #     intel_rss_longitudinal_min_safe_distance = 2.25  # meters
    #     longitudinal_distance = target_location.x - current_location.x
    #     if 0 < longitudinal_distance < intel_rss_longitudinal_min_safe_distance:
    #         return True
    #
    #     forward_vector = np.array([math.cos(math.radians(current_orientation)), math.sin(math.radians(current_orientation))])
    #     d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))
    #     return d_angle < 3
