#!/usr/bin/env python
from itertools import islice, cycle

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

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

LOOKAHEAD_WPS = 20  # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.waypoints = None
        self.current_pos = None

        rospy.spin()

    def pose_cb(self, msg):
        self.current_pos = msg.pose
        rospy.loginfo("WaypointUpdater: current car position %s", self.current_pos)
        self.publish_waypoints()

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        rospy.loginfo("WaypointUpdater: received waypoints")
        self.publish_waypoints()

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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

        final_waypoints = list(islice(cycle(self.waypoints), nearest_wp_idx, nearest_wp_idx + LOOKAHEAD_WPS - 1))

        lane = Lane()
        lane.waypoints = final_waypoints
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)

        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
