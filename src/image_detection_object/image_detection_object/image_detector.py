# ROS2
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


# image processing 

import cv2
import mediapipe as mp
import time
import threading

# MediaPipe Object Detector
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# custom class

from gesture_interpreter import HandDetector
from object_interpreter import ObjectDetectorWrapper


# Store the latest detection results
latest_result = None
result_lock = threading.Lock()

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/video_frames/compressed',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.hand_detector = HandDetector()
        self.object_detector = ObjectDetectorWrapper()
        self.cv_bridge = CvBridge()
        self.string_publisher = self.create_publisher(
            String,
            'robot_functions',
            10
        )

    def listener_callback(self, msg):
        try:
            print("recv: image")
            # self.get_logger().info('I heard: "%s"' % msg.data)
            publish = "None"
            image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            result = self.hand_detector.run_detection(image)
            # print(f"hand result = {result}")
            if result == "None":
                result = self.object_detector.run_detection(image)
                print(f"detected Object = {result}")
                        # Publish the string message onto ROS
            
            pub_msg = String()
            pub_msg.data = str(result)
            self.string_publisher.publish(pub_msg)

        except Exception as e:
            print(e)
        


def main(args=None):
    rclpy.init(args=args)
    
    image_handler_sub = ImageSubscriber()
    
    try:
        rclpy.spin(image_handler_sub)
    except:
        pass
    finally:
        image_handler_sub.destroy_node()


if __name__ == '__main__':
    main()