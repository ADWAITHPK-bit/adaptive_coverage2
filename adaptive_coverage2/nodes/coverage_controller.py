#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from tf_transformations import euler_from_quaternion

from adaptive_coverage2.geometry.compute_centroid import compute_centroid
from adaptive_coverage2.geometry.compute_laplacian import laplacian_consensus
from adaptive_coverage2.geometry.adaptive_law import AdaptiveEstimator


# ---------- helpers ----------
def clamp_position(p, xmin, xmax, ymin, ymax):
    p[0, 0] = np.clip(p[0, 0], xmin, xmax)
    p[1, 0] = np.clip(p[1, 0], ymin, ymax)
    return p


def project_velocity(position, u, xmin, xmax, ymin, ymax):
    x, y = position.flatten()

    if x <= xmin and u[0, 0] < 0:
        u[0, 0] = 0.0
    if x >= xmax and u[0, 0] > 0:
        u[0, 0] = 0.0
    if y <= ymin and u[1, 0] < 0:
        u[1, 0] = 0.0
    if y >= ymax and u[1, 0] > 0:
        u[1, 0] = 0.0

    return u


class CoverageController(Node):

    def __init__(self):
        super().__init__('coverage_controller')

        # -------- PARAMETERS --------
        self.declare_parameter('robot_name', 'tb3_1')
        self.robot_name = self.get_parameter('robot_name').value

        self.kp = 1.0
        self.gamma = 0.5
        self.zeta = 0.1
        self.dt = 0.1

        # Workspace bounds
        self.X_MIN, self.X_MAX = -5.0, 5.0
        self.Y_MIN, self.Y_MAX = -5.0, 5.0

        # -------- STATE --------
        self.position = None
        self.yaw = 0.0

        self.n_basis = 9
        self.a_hat = np.ones((self.n_basis, 1)) * 0.1
        self.estimator = AdaptiveEstimator(self.n_basis)

        self.neighbor_positions = {}
        self.neighbor_a_hat = {}
        self.l_ij = {}

        # -------- ROS --------
        self.create_subscription(
            Odometry,
            f'/{self.robot_name}/odom',
            self.odom_cb,
            10
        )

        self.create_subscription(
            Float64MultiArray,
            f'/{self.robot_name}/neighbor_positions',
            self.neighbor_pos_cb,
            10
        )

        self.create_subscription(
            Float64MultiArray,
            f'/{self.robot_name}/neighbor_a_hat',
            self.neighbor_param_cb,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            f'/{self.robot_name}/cmd_vel',
            10
        )

        self.param_pub = self.create_publisher(
            Float64MultiArray,
            f'/{self.robot_name}/a_hat',
            10
        )

        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(f"Coverage controller started for {self.robot_name}")

    # -------- callbacks --------
    def odom_cb(self, msg):
        self.position = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y]
        ])

        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

    def neighbor_pos_cb(self, msg):
        self.neighbor_positions.clear()
        self.l_ij.clear()

        data = msg.data
        for i in range(0, len(data), 3):
            rid = int(data[i])
            self.neighbor_positions[f'r{rid}'] = [data[i+1], data[i+2]]
            self.l_ij[f'r{rid}'] = 1.0

    def neighbor_param_cb(self, msg):
        self.neighbor_a_hat.clear()
        data = msg.data

        for i in range(0, len(data), self.n_basis + 1):
            rid = int(data[i])
            self.neighbor_a_hat[f'r{rid}'] = np.array(
                data[i+1:i+1+self.n_basis]
            ).reshape(self.n_basis, 1)

    # -------- density --------
    def true_density(self, p):
        x, y = p.flatten()
        return np.exp(-((x - 1.0)**2 + (y - 1.0)**2))

    # -------- main loop --------
    def control_loop(self):
        if self.position is None:
            return

        # --- centroid ---
        c_i, phi_i, _ = compute_centroid(
            self.position,
            self.a_hat,
            self.neighbor_positions
        )

        # Clamp centroid (CRITICAL)
        c_i = clamp_position(
            c_i,
            self.X_MIN, self.X_MAX,
            self.Y_MIN, self.Y_MAX
        )

        c_i[0,0] = np.clip(c_i[0,0], self.X_MIN + 0.2, self.X_MAX - 0.2)
        c_i[1,0] = np.clip(c_i[1,0], self.Y_MIN + 0.2, self.Y_MAX - 0.2)

        # --- cartesian control ---
        u = -self.kp * (self.position - c_i)

        # --- workspace projection ---
        u = project_velocity(
            self.position, u,
            self.X_MIN, self.X_MAX,
            self.Y_MIN, self.Y_MAX
        )

        # --- hard safety: stop if outside ---
        x, y = self.position.flatten()
        if x < self.X_MIN or x > self.X_MAX or y < self.Y_MIN or y > self.Y_MAX:
            u[:] = 0.0

        # --- unicycle mapping ---
        ux, uy = u.flatten()
        theta = self.yaw

        v = np.cos(theta) * ux + np.sin(theta) * uy
        w = -np.sin(theta) * ux + np.cos(theta) * uy

        v = np.clip(v, 0.0, 0.4)
        w = np.clip(w, -1.5, 1.5)

        # --- adaptive law ---
        lam = self.true_density(self.position)
        if phi_i is not None:
            self.estimator.update_memory(phi_i, lam, self.dt)

        a_cons = laplacian_consensus(
            self.a_hat, self.neighbor_a_hat, self.l_ij, self.zeta
        )

        a_dot = -self.gamma * (
            self.estimator.Lambda @ self.a_hat -
            self.estimator.lambda_vec
        ) + a_cons

        self.a_hat = self.estimator.projection(
            self.a_hat + a_dot * self.dt
        )

        # --- publish ---
        msg = Float64MultiArray()
        msg.data = self.a_hat.flatten().tolist()
        self.param_pub.publish(msg)

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CoverageController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
