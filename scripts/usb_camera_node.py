#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Empty
import sys


class USBCameraNode:
    def __init__(self):
        rospy.init_node('usb_camera_node')
        self.bridge = CvBridge()
        
        self.img_pub = rospy.Publisher('/camera/color/image_raw/compressed', CompressedImage, queue_size=10)

        self.camera_device = rospy.get_param('~device_id', 4)

        # if len(sys.argv) > 1 and sys.argv[1].isdigit():
        #     self.camera_device = int(sys.argv[1])
        # else:
        # self.camera_device = 4
        rospy.loginfo(f"Cam port is {self.camera_device}.")
        # self.camera_device = "/dev/usb_camera"
        self.cap = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            rospy.logerr(f"Can't open the camera device: {self.camera_device}")
            sys.exit(1)
            
        # 参数设置
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)


    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("Read frame failed.")
                rate.sleep()
                continue
            
            # 转换为ROS消息
            try:
                img_msg = self.bridge.cv2_to_compressed_imgmsg(frame, "jpg")
                img_msg.header.stamp = rospy.Time.now()
                self.img_pub.publish(img_msg)
                    
            except Exception as e:
                rospy.logerr(f"Convert image failed: {str(e)}")
            
            rate.sleep()

    def __del__(self):
        """释放摄像头资源"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            print("USB camera node stopped.")

if __name__ == '__main__':
    cam = USBCameraNode()
    rospy.loginfo("\033[32mUSB camera node started.\033[0m")
    cam.run()

