import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray,PointStamped
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

from skimage.morphology import erosion, disk, dilation
from skimage.measure import block_reduce
import numpy as np
import heapq
import pdb
import time
import dubins
from tf_transformations import euler_from_quaternion

from .a_star import *


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        # self.resolution = 0.5
        # self.step_size = self.resolution
        self.map = None
        self.start_pose = None
        self.end_pose = None

        self.start_theta = None
        self.end_theta = None

        self.return_start = None
        self.return_end = None

        self.RESOLUTION = 0.0504
        self.POOL_SIZE = 5
        self.EROSION_SIZE = 7
        self.DILATION_SIZE = 10
        self.TURNING_RAD = 0.848
        self.THRESHOLD_ANGLE = 3.141

        with open('path_test.txt', 'w') as file:
            pass

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")


    def map_cb(self, msg):
        """
        Takes in an OccupancyGrid msg and creates internal representation of the map in the form of
        a 2 dimensional np array defining the grid space.
        Params: 
        msg-OccupancyGrid message
        returns: 
        None
        """
        cols = msg.info.width
        rows = msg.info.height
        pool_size = (self.POOL_SIZE, self.POOL_SIZE)
        # map = erode_map(np.array(msg.data).reshape((rows, cols)), .4/.0504)
        # map = dilate_map(erode_map(np.array(msg.data).reshape((rows, cols)), self.EROSION_SIZE), self.DILATION_SIZE)
        map = erode_map(dilate_map(np.array(msg.data).reshape((rows, cols)), self.DILATION_SIZE), self.EROSION_SIZE)
        self.map = block_reduce(map, pool_size, np.max)

        
        self.get_logger().info(f"got map, shape: {self.map.shape}")

    def pose_cb(self, pose):
        """
        Sets initial pose for path planning and starts path planning if goal pose and map is set.
        Resets pose after path planning
        params:
        pose - PoseWithCovarianceStamped
        """

        self.start_pose = pose
        self.get_logger().info("got initial pose")
        # self.get_logger().info(f'{self.end_pose} {self.map}')
        if self.end_pose is not None and self.map is not None:
            self.trajectory.clear()
            self.final_path(self.start_pose, self.end_pose, self.map)
            self.start_pose = None
            self.end_pose = None
            self.get_logger().info("ending initial pose cb")

    def goal_cb(self, msg):
        """
        Sets goal pose for path planning and starts path planning if initial pose and map is set.
        Resets poses after path planning.
        """
        self.end_pose = msg
        self.get_logger().info("got goal pose")
        # self.get_logger().info(f'{self.start_pose} {self.map}')
        if self.start_pose is not None and self.map is not None:
            self.trajectory.clear()
            self.final_path(self.start_pose, self.end_pose, self.map)
            self.start_pose = None
            self.end_pose = None
            self.get_logger().info("ending goal pose cb")

    def transformations(self, start_pose, end_pose):
        self.get_logger().info("transforming")

        quaternion = start_pose.pose.pose.orientation
        # Convert the quaternion to Euler angles (roll, pitch, yaw)
        (_, _, yaw) = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.start_theta = yaw

        quaternion = end_pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.end_theta = yaw

        # may have to deal with downsampling here too
        self.return_start = transform_wtm(start_pose.pose.pose.position.x, start_pose.pose.pose.position.y, self.start_theta)
        start = (int(self.return_start[0]/self.POOL_SIZE), int(self.return_start[1]/self.POOL_SIZE))
        self.get_logger().info(f"start point {start[0],start[1]}: {self.map[int(start[1]),int(start[0])]}")
        
        self.return_end = transform_wtm(end_pose.pose.position.x, end_pose.pose.position.y, self.end_theta)
        end = (int(self.return_end[0]/self.POOL_SIZE), int(self.return_end[1]/self.POOL_SIZE))
        self.get_logger().info(f"end point {end[0],end[1]}: {self.map[int(end[1]),int(end[0])]}")

        return start, end

    def final_path(self, start_pose, end_pose, map):
        '''
        plans a path in three parts:
        #1 going from start point to nearest point on center line
        #2 taking the trajectory of center line and going until point that is closest to end point on trajectory (probably with lookahead)
        #3 going from center line to final point

        if backwards:
            step 1 becomes making a uturn

        '''
        # STEP 1:
        # TODO: need to get the trajectory of line
        # TODO: calculate the closest point with perp line (do we want to offset this a little)

        start, end = self.transformations(start_pose, end_pose)

        # POINTS ARE TRANSFORMED INTO CORRECT SCALE AND DOWNSAMPLED PAST HERE

        a_star_1 = AStarNode(self.return_start, self.return_end, map)
        path_1 = a_star_1.plan_path(start, end)

        self.publish_trajectory(path_1)

        raise NotImplementedError



    def publish_trajectory(self, path):
        if path:
            self.get_logger().info("path planned woohoo")
            for coords in path:
                # made the coords floats cuz it was complaining
                coords = (float(coords[0]), float(coords[1]))
                self.trajectory.addPoint(coords)         
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
        else:
            self.get_logger().info("no path, sad")

    


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
