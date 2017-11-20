from styx_msgs.msg import TrafficLight

import numpy as np
import os
import tensorflow as tf
from object_detection.utils import label_map_util
import rospy
import time

class TLClassifier(object):
    def __init__(self, simulator=True):

        self.simulator = simulator
        self.current_light = TrafficLight.UNKNOWN
        cwd = os.path.dirname(os.path.realpath(__file__))

        # load different graphs dep on config, sim or real
        if self.simulator:
            CKPT = cwd + '/graphs/sim/frozen_inference_graph.pb'
        else:
            CKPT = cwd + '/graphs/real/frozen_inference_graph.pb'

        label_map = label_map_util.load_labelmap(
            cwd + '/graphs/label_map.pbtxt')
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=14,
                                                                    # 14 from bosch dataset
                                                                    use_display_name=True)

        # get label map categories
        self.category_index = label_map_util.create_category_index(categories)

        # TF graph set up
        self.classifer_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.classifer_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.classifer_graph, config=config)

        self.image_tensor = self.classifer_graph.get_tensor_by_name(
            'image_tensor:0')
        self.detection_boxes = self.classifer_graph.get_tensor_by_name(
            'detection_boxes:0')
        self.detection_scores = self.classifer_graph.get_tensor_by_name(
            'detection_scores:0')
        self.detection_classes = self.classifer_graph.get_tensor_by_name(
            'detection_classes:0')
        self.num_detections = self.classifer_graph.get_tensor_by_name(
            'num_detections:0')

        rospy.loginfo("*** TF classifer loaded! ***")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        WORK IN PROGRESS
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        tic = time.time()
        self.current_light = TrafficLight.UNKNOWN
        image_np_expanded = np.expand_dims(image, axis=0)

        # run detection
        with self.classifer_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # light color prediction
        min_score_thresh = .50
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                class_name = self.category_index[classes[i]]['name']
                if class_name == 'Red':
                    self.current_light = TrafficLight.RED
                elif class_name == 'Green':
                    self.current_light = TrafficLight.GREEN
                elif class_name == 'Yellow':
                    self.current_light = TrafficLight.YELLOW

        toc = time.time()
        rospy.loginfo('classifier took {} sec'.format(toc-tic))
        return self.current_light
