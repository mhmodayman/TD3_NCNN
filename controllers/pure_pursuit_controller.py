#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""
import math

import numpy as np
from collections import deque


class PurePursuitController(object):
    def __init__(self, waypoints, dt, wheelbase, v_target):
        self.v_previous = 0.0
        self.error_previous = 0.0
        self.integral_error_previous = 0.0
        self.throttle_previous = 0.0
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._target_speed = v_target
        self._current_frame = 0
        self._start_control_loop = False
        self._set_throttle = 0
        self._set_brake = 0
        self._set_steer = 0
        self._waypoints = waypoints
        self.max_steering_angle = 70.0  # [deg] retrieved from carla simulator for vehicle audi a2
        self._conv_rad_to_steer = 180.0 / self.max_steering_angle / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi
        self.dt = dt

        self.k = 0.1  # look forward gain
        self.Lfc = 1.0  # look-ahead distance
        self._ego_wheelbase = wheelbase

        # parameters for pid speed controller
        self.e_buffer = deque(maxlen=20)
        self._e = 0
        self.K_P = 1.0
        self.K_D = 0.2
        self.K_I = 0.01

    def update_values(self, x, y, yaw, speed, frame):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed
        self._current_frame = frame
        # doesn't apply at time step 0, control loop skips first frame to store prev values
        if self._current_frame:
            self._start_control_loop = True

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Convert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        v_target = self._target_speed
        waypoints = self._waypoints
        throttle_output = 0
        steer_output = 0
        brake_output = 0
        e_v = 0
        inte_v = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
            Controller iteration code block.

            Controller Feedback Variables:
                x               : Current X position (meters)
                y               : Current Y position (meters)
                yaw             : Current yaw pose (radians)
                v               : Current forward speed (meters per second)
                t               : Current time (seconds)
                v_target       : Current desired speed (meters per second)
                                  (Computed as the speed to track at the
                                  closest waypoint to the vehicle.)
                waypoints       : Current waypoints to track
                                  (Includes speed to track at each x,y
                                  location.)
                                  Format: [[x0, y0, v0],
                                           [x1, y1, v1],
                                           ...
                                           [xn, yn, vn]]
                                  Example:
                                      waypoints[2][1]: 
                                      Returns the 3rd waypoint's y position

                                      waypoints[5]:
                                      Returns [x5, y5, v5] (6th waypoint)

            Controller Output Variables:
                throttle_output : Throttle output (0 to 1)
                steer_output    : Steer output (-1.22 rad to 1.22 rad)
                brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################

            self._e = v_target - v
            self.e_buffer.append(self._e)

            if len(self.e_buffer) >= 2:
                _de = (self.e_buffer[-1] - self.e_buffer[-2]) / self.dt
                _ie = sum(self.e_buffer) * self.dt
            else:
                _de = 0.0
                _ie = 0.0

            throttle_output = np.clip((self.K_P * self._e) + (self.K_D * _de / self.dt) + (self.K_I * _ie * self.dt), -0.35, 0.35)

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################

            # Change the steer output with the lateral controller.
            steer_output = 0

            # search nearest point index
            length = np.arange(0, waypoints.shape[0], 1)
            dx = [self._current_x - waypoints[icx][0] for icx in length]
            dy = [self._current_y - waypoints[icy][1] for icy in length]
            d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
            ind = d.index(min(d))
            if ind < 2:
                tx = waypoints[ind][0]
                ty = waypoints[ind][1]
            else:
                tx = waypoints[-1][0]
                ty = waypoints[-1][1]
                # ind = len(self._current_x) - 1

            alpha_hat = math.atan2(ty - y, tx - x)
            alpha = alpha_hat - yaw
            Lf = self.k * v + self.Lfc

            # final expected steering
            steer_expect = math.atan2(2.0 * self._ego_wheelbase * math.sin(alpha) / Lf, 1.0)
            if steer_expect > np.pi:
                steer_expect -= 2 * np.pi
            if steer_expect < - np.pi:
                steer_expect += 2 * np.pi
            steer_expect = min(1.22, steer_expect)
            steer_expect = max(-1.22, steer_expect)

            # update
            steer_output = steer_expect

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)  # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)  # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.v_previous = v  # Store forward speed to be used in next step
        self.throttle_previous = throttle_output
        self.error_previous = e_v
        self.integral_error_previous = inte_v
