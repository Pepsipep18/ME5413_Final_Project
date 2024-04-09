#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import String
from vision_msgs.msg import Detection2D
import os

template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "124_160.png")

class TargetDetection(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/front/rgb/image_raw", Image, self.input_img_callback)
        self.depth_image_sub = rospy.Subscriber('/front/depth/image_raw', Image, self.depth_callback)
        self.template_sub = rospy.Subscriber("/rviz_panel/goal_name", String, self.template_callback)

        self.template = None
        self.ori_coodinate = (0, 0, 124, 160)
        self.template_path = template_path
        self.done = False

        self.depth_image = None

        self.detected_pub = rospy.Publisher("/me5413/detection", Detection2D, queue_size=10)
        self.current_depth_pub = rospy.Publisher("/me5413/depth", Image, queue_size=10)

    def template_callback(self, data):
        if "box" not in data.data:
            self.template = None
            return

        try:
            img = cv2.imread(self.template_path)
            if img is None:
                rospy.logerr("Failed to load image")
            else:
                x, y, w, h = self.ori_coodinate
                self.template = img[y:y+h, x:x+w]

        except Exception as e:
            rospy.logerr("Error loading template image: {}".format(e))

    def depth_callback(self, data):
        self.depth_image = data

    def input_img_callback(self, data):
        if self.template is None:
            detection = Detection2D()
            detection.bbox.size_x = 0
            detection.bbox.size_y = 0
            detection.bbox.center.x = 0
            detection.bbox.center.y = 0
            detection.source_img = data
            self.detected_pub.publish(detection)
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "8UC3")
            current_depth = self.depth_image
            self.detection(cv_image, current_depth)
        except CvBridgeError as e:
            print(e)

    def publish_detection(self, x, y, w, h, img, current_depth):
        detection = Detection2D()

        detection.bbox.size_x = w
        detection.bbox.size_y = h
        detection.bbox.center.x = x + w // 2
        detection.bbox.center.y = y + h // 2

        try:
            ros_img = self.bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        detection.source_img = ros_img

        self.detected_pub.publish(detection)
        self.current_depth_pub.publish(current_depth)

    def detection(self, image, current_depth):

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = image_gray
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        processed_template = template_gray

        height, width = processed_template.shape[:2]

        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        max_match_val = -1
        best_scale = None
        best_max_loc = None

        for s in scales:
            new_width = int(width * s)
            new_height = int(height * s)

            resized_template = cv2.resize(processed_template, (new_width, new_height),
                                          interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)

            result = cv2.matchTemplate(processed_image, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > max_match_val:
                rospy.loginfo("Max value: {}".format(max_val))
                max_match_val = max_val
                best_scale = s
                best_max_loc = max_loc

        if max_match_val > 0.75:
            x, y = best_max_loc
            w = int(width * best_scale)
            h = int(height * best_scale)
        else:
            x, y, w, h = 0, 0, 0, 0

        self.publish_detection(x, y, w, h, image, current_depth)
        return image



rospy.init_node('target_detection', anonymous=True)
det = TargetDetection()
rospy.spin()

