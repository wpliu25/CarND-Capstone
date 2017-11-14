#!/usr/bin/env python
from itertools import islice, cycle

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 35  # Number of waypoints we will publish. You can change this number
ACCELERATION_MAX = 1.0  # copied from waypoint_loader


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # check if obstacle is needed?
        #rospy.Subscriber('/obstacle_waypoints', '', self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.waypoints = None
        self.current_pos = None
        self.current_wp = None
        self.final_waypoints = None
        self.updated_velocity_flag = False
        self.traffic_light_wp = -1

        rospy.spin()

    def pose_cb(self, msg):
        self.current_pos = msg.pose
        #rospy.loginfo("WaypointUpdater: current car position %s", self.current_pos)
        self.publish_waypoints()

    def waypoints_cb(self, waypoints):
        if self.waypoints:
            return

        self.waypoints = waypoints.waypoints
        rospy.loginfo("WaypointUpdater: received waypoints")
        self.publish_waypoints()

    def calc_time_between_two_wps(self, wp1, wp2):
        dist = self.distance(self.waypoints, wp1, wp2)
        time = math.sqrt(2.0*dist/ACCELERATION_MAX)

        return time

    def calc_velocity_between_two_wps(self, wp2_velocity, time):
        return wp2_velocity+ACCELERATION_MAX*time

    def euclidean_distance(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)

    def decelerate(self, waypoints, stop_idx):
        last = waypoints[stop_idx]
        last.twist.twist.linear.x = 0.
        for i, wp in enumerate(waypoints):
            if i < stop_idx:
                dist = self.euclidean_distance(wp.pose.pose.position, last.pose.pose.position)
                vel = math.sqrt(2.0 * ACCELERATION_MAX * dist)
                if vel < 1.:
                    vel = 0.
                wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            else:
                wp.twist.twist.linear.x = 0.0

        return waypoints

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.traffic_light_wp = int(msg.data)
        # if self.traffic_light_wp < self.current_wp:
        #     self.traffic_light_wp = -1

        self.publish_waypoints()

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def publish_waypoints(self):
        if self.waypoints is None or self.current_pos is None:
            return

        car_x = self.current_pos.position.x
        car_y = self.current_pos.position.y

        nearest_wp_idx = None
        nearest_distance = None

        for i in range(len(self.waypoints)):
            wp = self.waypoints[i]
            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y

            distance = math.sqrt((car_x - wp_x)**2 + (car_y - wp_y)**2)

            if nearest_distance is None:
                nearest_distance = distance
                nearest_wp_idx = i
            else:
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_wp_idx = i

        self.current_wp = nearest_wp_idx

        final_waypoints = list(islice(cycle(self.waypoints), nearest_wp_idx, nearest_wp_idx + LOOKAHEAD_WPS))

        #rospy.loginfo("current_wp %s tl_wp %s", self.current_wp, self.traffic_light_wp)

        speed_up = True
        if self.traffic_light_wp != -1:  # red traffic light
            if (self.traffic_light_wp - self.current_wp) < LOOKAHEAD_WPS:
                self.decelerate(final_waypoints, self.traffic_light_wp - self.current_wp - 6)
                speed_up = False

        if speed_up:
            velocity = self.kmph2mps(rospy.get_param('~velocity'))
            if final_waypoints is not None:
                for wp in final_waypoints:
                    wp.twist.twist.linear.x = velocity

        lane = Lane()
        lane.waypoints = final_waypoints
        #for i in range(0, 10):
        #rospy.loginfo("publish_waypoints current velocity %s", self.final_waypoints[0].twist.twist.linear.x)

        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)

        # save current wp
        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
