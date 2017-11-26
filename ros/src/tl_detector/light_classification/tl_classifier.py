from styx_msgs.msg import TrafficLight

import numpy as np
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils
import rospy
import time
import cv2

import sys
from functools import partial
def join_graph(directory, filename, chunksize=1024):
    print "Restoring:", filename, "\nfrom directory:", directory
    if os.path.exists(directory):
        if os.path.exists(filename):
            os.remove(filename)
        output = open(filename, 'wb')
        chunks = os.listdir(directory)
        chunks.sort()
        for fname in chunks:
            fpath = os.path.join(directory, fname)
            with open(fpath, 'rb') as fileobj:
                for chunk in iter(partial(fileobj.read, chunksize), ''):
                    output.write(chunk)
        output.close()


class TLClassifier(object):
    def __init__(self, debug=False, simulator=False):
        
        self.debug = debug

        if self.debug:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 500, 500)

        self.simulator = simulator
        self.current_light = TrafficLight.UNKNOWN
        cwd = os.path.dirname(os.path.realpath(__file__))

        # load different graphs dep on config, sim or real
        if self.simulator:
            CKPT = cwd + '/graphs/sim/frozen_inference_graph.pb'
            if not os.path.exists(CKPT):
                join_graph( (cwd + '/graphs/sim/frozen_inference_graph_chunks'), CKPT)

        else:
            CKPT = cwd + '/graphs/real/frozen_inference_graph.pb'
            if not os.path.exists(CKPT):
                join_graph( (cwd + '/graphs/real/frozen_inference_graph_chunks'), CKPT)

        label_map = label_map_util.load_labelmap(
            cwd + '/graphs/label_map.pbtxt')
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=14,
                                                                    # 14 from bosch dataset
                                                                    use_display_name=True)

        # get label map categories
        self.category_index = label_map_util.create_category_index(categories)
        print self.category_index

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

        rospy.loginfo("*** TF classifer {} loaded! ***".format(CKPT))


    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        WORK IN PROGRESS
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if self.debug:
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

        if self.debug:
            rospy.loginfo('classes: %s \n scores %s ' % (classes[:5], scores[:5]))

        # light color prediction
        min_score_thresh = .3
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                class_name = self.category_index[classes[i]]['name']
                if class_name == 'Red':
                    self.current_light = TrafficLight.RED
                elif class_name == 'Green':
                    self.current_light = TrafficLight.GREEN
                elif class_name == 'Yellow':
                    self.current_light = TrafficLight.YELLOW
                break # Here we go we found best match!

        # Visualization of the results of a detection.
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image, boxes, classes, scores,
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)  # For visualization topic output
        if self.debug:
            self._current_image = image
            cv2.imshow('image', image)
            cv2.waitKey(1)

        if self.debug:
            toc = time.time()
            rospy.loginfo('classifier took {} sec'.format(toc-tic))
        return self.current_light
