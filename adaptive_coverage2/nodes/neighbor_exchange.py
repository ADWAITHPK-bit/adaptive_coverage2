#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

class NeighborExchange(Node):

    def __init__(self):
        super().__init__('neighbor_exchange')

        self.declare_parameter('robot_id', 1)
        self.declare_parameter('robot_ids', [1,2])

        self.my_id = self.get_parameter('robot_id').value
        self.ids = self.get_parameter('robot_ids').value

        self.positions = {}
        self.a_hats = {}

        for i in self.ids:
            if i == self.my_id:
                continue

            self.create_subscription(
                Odometry, f'/tb3_{i}/odom',
                lambda m, rid=i: self.odom_cb(m,rid), 10)

            self.create_subscription(
                Float64MultiArray, f'/tb3_{i}/a_hat',
                lambda m, rid=i: self.param_cb(m,rid), 10)

        self.pos_pub = self.create_publisher(
            Float64MultiArray,
            f'/tb3_{self.my_id}/neighbor_positions',10)

        self.param_pub = self.create_publisher(
            Float64MultiArray,
            f'/tb3_{self.my_id}/neighbor_a_hat',10)

        self.create_timer(0.1, self.publish)

    def odom_cb(self, msg, rid):
        if rid != self.my_id:
            self.positions[rid] = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y
            ]

    def param_cb(self, msg, rid):
        if rid != self.my_id:
            self.a_hats[rid] = msg.data

    def publish(self):
        pmsg = Float64MultiArray()
        for rid,p in self.positions.items():
            pmsg.data.extend([rid,p[0],p[1]])

        amsg = Float64MultiArray()
        for rid,a in self.a_hats.items():
            amsg.data.append(rid)
            amsg.data.extend(a)

        self.pos_pub.publish(pmsg)
        self.param_pub.publish(amsg)


def main(args=None):
    rclpy.init(args=args)
    node = NeighborExchange()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
