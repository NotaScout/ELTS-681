import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')

        self.publisher_ = self.create_publisher(
            CompressedImage,
            '/video_frames/compressed',
            10
        )

        self.cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError('Could not open camera at /dev/video0')

        self.br = cv2
        self.timer = self.create_timer(0.1, self.publish_images)

        self.get_logger().info(
            'Publishing compressed grayscale images on /video_frames/compressed'
        )

    def publish_images(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning('Failed to capture image')
            return

        # Convert BGR camera frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compress grayscale image to JPEG
        success, encoded = cv2.imencode(
            '.jpg',
            gray,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if not success:
            self.get_logger().warning('JPEG compression failed')
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_optical_frame'
        msg.format = 'mono8; jpeg'
        msg.data = encoded.tobytes()
        
        self.publisher_.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = ImagePublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError) as e:
        if str(e):
            print(e)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()