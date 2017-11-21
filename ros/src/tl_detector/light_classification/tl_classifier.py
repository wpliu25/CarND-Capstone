from styx_msgs.msg import TrafficLight

import numpy as np
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from darkflow.net.build import TFNet
import rospy
import time

class TLClassifier(object):
    def __init__(self, simulator=True):

        self.tryDarkflow = False

        self.simulator = simulator
        self.current_light = TrafficLight.UNKNOWN
        cwd = os.path.dirname(os.path.realpath(__file__))

        if not self.tryDarkflow:

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

        else:

            use_gpu = False;

            cwd = os.path.dirname(os.path.realpath(__file__))

            options = {
                       "summary": None}

            if self.simulator:
                options["model"] = os.path.join(cwd, "cfg/sim_data.cfg")
                options["metaload"] = os.path.join(cwd, "ckpt/sim_data-34500.meta")
                options["labels"] = os.path.join(cwd, "ckpt/labels_sim.txt")
            else:
                options["model"] = os.path.join(cwd, "cfg/yolo_bosch.cfg")
                options["metaload"] = os.path.join(cwd, "ckpt/yolo_bosch-80000.meta")
                options["labels"] = os.path.join(cwd, "ckpt/labels_real.txt")
                options["threshold"] = 0.1

            if use_gpu:
                options["gpu"] = 1.0

            rospy.loginfo("YOLO classifier loaded")

            self.tfnet = TFNet(options)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        WORK IN PROGRESS
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        tic = time.time()

        if not self.tryDarkflow:

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

        else:

            result = self.tfnet.return_predict(image)

            rospy.loginfo(result)

            if not result:
                self.current_light = TrafficLight.UNKNOWN

            # extract results

            red     = [l for l in result if l["label"] == "Red"]
            green   = [l for l in result if l["label"] == "Green"]
            yellow  = [l for l in result if l["label"] == "Yellow"]
            off     = [l for l in result if l["label"] == "off"]

            # definitely break if there is any red light
            if red:
                self.current_light = TrafficLight.RED

            if yellow:
                self.current_light = TrafficLight.YELLOW

            if green:
                self.current_light = TrafficLight.GREEN

        toc = time.time()
        rospy.loginfo('classifier took {} sec'.format(toc-tic))
        return self.current_light
