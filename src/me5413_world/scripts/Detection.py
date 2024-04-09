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
        self.template_sub = rospy.Subscriber("/rviz_panel/goal_name", String, self.template_callback)

        self.template = None
        self.ori_coodinate = (0, 0, 124, 160)
        self.template_path = template_path

        self.detected_pub = rospy.Publisher("/me5413/detection", Detection2D, queue_size=10)

    def template_callback(self, data):
        # check whether click the image
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
            rospy.logerr("Cannot load image")


    def input_img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "8UC3")
            self.detection(cv_image)
        except CvBridgeError as e:
            rospy.loginfo_once("Input image callback")

    def publish_detection(self, x, y, w, h, img):
        detection = Detection2D()

        detection.bbox.size_x = w
        detection.bbox.size_y = h
        detection.bbox.center.x = x + w // 2
        detection.bbox.center.y = y + h // 2

        try:
            ros_img = self.bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            print("publish_detection error")

        detection.source_img = ros_img

        self.detected_pub.publish(detection)

    def detection(self, image):

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_temp = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        height, width = gray_temp.shape[:2]

        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        threshold = 0
        scale = None
        best_location = None

        for s in scales:
            new_width = int(width * s)
            new_height = int(height * s)

            shaped_temp = cv2.resize(gray_temp, (new_width, new_height),
                                          interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)

            result = cv2.matchTemplate(gray_img, shaped_temp, cv2.TM_CCOEFF_NORMED)
            _, value, _, location = cv2.minMaxLoc(result)

            if value > threshold:
                rospy.loginfo("Max value: {}".format(value))
                threshold = value
                scale = s
                best_location = location

        if threshold > 0.75:
            x, y = best_location
            w = int(width * scale)
            h = int(height * scale)
        else:
            x, y, w, h = 0, 0, 0, 0

        self.publish_detection(x, y, w, h, image)
        return image



rospy.init_node('target_detection', anonymous=True)
det = TargetDetection()
rospy.spin()

