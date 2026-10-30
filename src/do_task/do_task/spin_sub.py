import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)

        self.wheel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        self.ready = True
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        
        if not self.ready:
            return
        
        if(msg.data == "Open Palm"):
            self.ready = False
            move = Twist()
            move.linear.x = 0.1
            move.angular.z = 0.0

            self.wheel_publisher.publish(move)
            # Stop the robot after 3 seconds
            self.timer = self.create_timer(3.0, self.stop_robot)

        elif(msg.data == "Fist"):
            self.ready = False
            move = Twist()
            move.linear.x = 0.0
            move.angular.z = 0.5

            self.wheel_publisher.publish(move)
            self.timer = self.create_timer(3.0, self.stop_robot)

        elif(msg.data == "Peace Sign"):
            self.ready = False
            move = Twist()
            move.linear.x = 0.0
            move.angular.z = 1.0

            self.wheel_publisher.publish(move)
            self.timer = self.create_timer(6.28, self.stop_robot)

        elif(msg.data == "Pointing"):
            self.ready = False
            move = Twist()
            move.linear.x = -0.1
            move.angular.z = 0.0

            self.wheel_publisher.publish(move)
            self.timer = self.create_timer(3.0, self.stop_robot)

        # elif(msg.data == "Hello World: 60"):
        #     move = Twist()
        #     move.linear.x = 0.0
        #     move.angular.z = 0.0
        
        #     self.wheel_publisher.publish(move)

    def stop_robot(self):
        move = Twist()
        move.linear.x = 0.0
        move.angular.z = 0.0

        self.wheel_publisher.publish(move)

        self.timer.cancel()
        self.ready = True



def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()