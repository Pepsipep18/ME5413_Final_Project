#!/usr/bin/env python
import rospy
import tf
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np


class CameraCalcuation:
    def __init__(self):
        rospy.init_node('camera_calcuation')

        self.listener = tf.TransformListener()

        self.camera_info_sub = rospy.Subscriber('/front/rgb/camera_info', CameraInfo, self.camera_params_callback)
        self.depth_image_sub = rospy.Subscriber('/front/depth/image_raw', Image, self.depth_callback)
        self.detection_sub = rospy.Subscriber('/me5413/detection', Detection2D, self.detection_callback)

        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        self.bridge = CvBridge()
        self.camera_info = None
        self.depth_image = None

    # get the camera parameters
    def camera_params_callback(self, params):
        self.camera_info = params

    def depth_callback(self, depth):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth, desired_encoding="32FC1")

    def detection_callback(self, detection):
        if self.camera_info is None or self.depth_image is None:
            rospy.loginfo("......No detection......")
            return

        if detection.bbox.size_x == 0 or detection.bbox.size_y == 0:
            return

        rospy.loginfo("Target detected.")
        bbox = detection.bbox
        center = bbox.center

        depth = self.depth_image[int(center.y), int(center.x)]

        if np.isnan(depth) or np.isinf(depth):
            rospy.loginfo("Invalid Depth.")
            return

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        # calculate the depth
        X = (center.x - cx) * depth / fx
        Y = (center.y - cy) * depth / fy
        Z = depth - 0.5 
        if Z < 0:
            Z = 0.01

        point_stamped = PointStamped()
        current_time = rospy.Time.now()
        point_stamped.header.stamp = current_time
        point_stamped.header.frame_id = "front_frame_optical"
        point_stamped.point.x = X
        point_stamped.point.y = Y
        point_stamped.point.z = Z

        self.listener.waitForTransform("map", point_stamped.header.frame_id, current_time, rospy.Duration(4.0))
        map_point = self.listener.transformPoint("map", point_stamped)

        goal_pose = PoseStamped()
        goal_pose.header.stamp = current_time
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = map_point.point.x
        goal_pose.pose.position.y = map_point.point.y
        goal_pose.pose.position.z = map_point.point.z
        goal_pose.pose.orientation.w = 1.0

        rospy.loginfo("Target position successfully calculated by template: x={}, y={}, z={}".format(
            goal_pose.pose.position.x,
            goal_pose.pose.position.y,
            goal_pose.pose.position.z
        ))

        self.goal_pub.publish(goal_pose)
        rospy.loginfo("Goal pose published successfully.")

        return

if __name__ == '__main__':
    node = CameraCalcuation()
    rospy.spin()
