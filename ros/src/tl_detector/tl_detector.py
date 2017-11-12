#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoint = None
        self.camera_image = None
        self.lights = []
        # self.light_waypoints = None

        self.waypoints = 0
        self.num_waypoints = 0
        self.stop_line_waypoints = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.num_waypoints = len(waypoints.waypoints)
        self.calc_stop_line_waypoints()

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = statecing_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement

        if self.waypoints is None or self.num_waypoints == 0:
            return None

        min_dist = 1e9
        index = None

        for k in xrange(self.num_waypoints):
            delta_x = (self.waypoints.waypoints[k].pose.pose.position.x - pose.position.x)
            delta_y = (self.waypoints.waypoints[k].pose.pose.position.y - pose.position.y)

            # this is actually dist_squared
            # but we can just use it as comparison
            dist = delta_x*delta_x + delta_y*delta_y
            if dist < min_dist:
                #rospy.loginfo("dx: %s, \t dy: %s", delta_x, delta_y)
                index = k
                min_dist = dist

        return index

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def calc_stop_line_waypoints(self):
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        stop_line_waypoints = []
        for stop_line_pos in stop_line_positions:
            p = Pose()
            p.position.x = stop_line_pos[0]
            p.position.y = stop_line_pos[1]
            idx = self.get_closest_waypoint(p)
            stop_line_waypoints.append((idx, p.position.x, p.position.y))

        # save this list
        self.stop_line_waypoints = stop_line_waypoints

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        car_position = None
        
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)
            # str = 'closest way point: %s'%car_position
            # rospy.loginfo(str)

        #TODO find the closest visible traffic light (if one exists)
        stop_line_waypoints_and_deltas = []
        if self.stop_line_waypoints is not None and car_position is not None:
            for t in self.stop_line_waypoints:
                wp_idx = t[0]
                x = t[1]
                y = t[2]

                if wp_idx is not None:
                    delta_idx = wp_idx - car_position
                    # unwrap index if the stop line is behind our car
                    # this is necessary when we are close to the largest way point number
                    # because every stop line will be behind the car, i.e., delta_idx < 0
                    if(delta_idx < 0):
                        delta_idx = self.num_waypoints + delta_idx

                    stop_line_waypoints_and_deltas.append((wp_idx, delta_idx, x, y))

                    # if(delta_idx < 50):
                    #     #light = 0
                    #     #state = 0
                    #     
                    #     #return idx, 0
            POS_THRESHOLD = 50.0
            COLOR_STR = ['RED', 'YELLOW', 'GREEN', 'NAN','UNKNOWN']

            if(len(stop_line_waypoints_and_deltas) > 0):
                # sort the list by delta_idx
                stop_line_waypoints_and_deltas.sort(key=lambda s:s[1])
                # choose the first/closest light
                closest_stop_line_waypoint_and_delta_idx = stop_line_waypoints_and_deltas[0]

                # use x, y position to look up light state
                light_idx = [i for i, v in enumerate(self.lights) \
                    if abs(v.pose.pose.position.x - x) < POS_THRESHOLD and abs(v.pose.pose.position.y - y) < POS_THRESHOLD]

                if len(light_idx) > 0:
                    color = self.lights[light_idx[0]].state
                    rospy.loginfo("light ahead: delta_idx = %s, %s", closest_stop_line_waypoint_and_delta_idx[1], COLOR_STR[color])


        # --- Why not this instead of the above?
        #
        # approaching_stop_light_waypoint = None
        # approaching_stop_light_color = None
        #
        # for stop_line_pos in stop_line_positions:
        #     stop_light_waypoint = self.get_closest_waypoint(stop_line_position)
        #     # Get color ID from styx_msgs/TrafficLight !!
        #     stop_light_color = 'Get color ID from styx_msgs/TrafficLight'
        #
        #     # Check first occurrance of waypoint corresponding to light that is beyond car's waypoint
        #     if stop_light_waypoint >= car_position:
        #         approaching_stop_light_waypoint = stop_light_waypoint
        #         approaching_light_color = stop_light_color



        if light:
            state = self.get_light_state(light)
            return light_wp, state
        #self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
