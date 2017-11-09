import rospy

from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MAX_SPEED = 30
MIN_SPEED = 1


class Controller(object):
    def __init__(self, *args, **kwargs):
        self.yaw_control = YawController(kwargs['wheel_base'], kwargs['steer_ratio'],
                                         kwargs['min_speed'], kwargs['max_lat_accel'],
                                         kwargs['max_steer_angle'])

        self.low_pass_filter = LowPassFilter(0.15, 0.8)

        self.throttle_control = PID(0.1, 0.1, MIN_SPEED, MAX_SPEED)

        self.last_run_ts = None

    def control(self, v, w, current_v, dbw_status):
        if self.last_run_ts is None:
            self.last_run_ts = rospy.get_time()
            return 0.0, 0.0, 0.0

        if not dbw_status:
            self.last_run_ts = rospy.get_time()
            return 0.0, 0.0, 0.0

        time_passed = rospy.get_time() - self.last_run_ts

        cross_track_error_for_v = v.x - current_v.x
        throttle = self.throttle_control.step(cross_track_error_for_v, time_passed)

        brake = 0.0

        if cross_track_error_for_v < 0:
            # it seems like we need to push brake
            brake = 1.0
            throttle = 0.0

        steer = self.yaw_control.get_steering(v.x, w.z, current_v.x)
        steer = self.low_pass_filter.filt(steer)

        return throttle, brake, steer
