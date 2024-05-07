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

        self.line_traj = {"points": [{"x": -19.99921417236328, "y": 1.3358267545700073}, {"x": -18.433984756469727, "y": 7.590575218200684}, 
                                    {"x": -15.413466453552246, "y": 10.617328643798828}, {"x": -6.186201572418213, "y": 21.114534378051758}, 
                                    {"x": -5.5363922119140625, "y": 25.662315368652344}, {"x": -19.717021942138672, "y": 25.677358627319336}, 
                                    {"x": -20.30797004699707, "y": 26.20694923400879}, {"x": -20.441822052001953, "y": 33.974945068359375}, 
                                    {"x": -55.0716438293457, "y": 34.07769775390625}, {"x": -55.30067825317383, "y": 1.4463690519332886}]}

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
        '''
        takes in the start pose and end pose in world frame and then transforms into map
        sets self. start and end theta and self.return start and end to the transformed points
        returns the int of the start and end
        '''
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
        A checking backwards going from start point to nearest point on center line
        B taking the trajectory of center line and going until point that is closest to end point on trajectory (probably with lookahead)
        C going from center line to final point

        if backwards:
            A becomes making a u-turn
        '''

        start, end = self.transformations(start_pose, end_pose)

        # POINTS ARE TRANSFORMED INTO CORRECT SCALE AND DOWNSAMPLED PAST HERE

        # STEP 1: checking if the line is backwards
        a_star = AStarNode(self.return_start, self.return_end, map)
        opt_path, backwards = a_star.plan_path()
        self.publish_trajectory(final_path)

        # TODO: need to get the trajectory of line MAKE SURE THIS IS IN MAP FRAME
        # load the trajectory, interpolate points in between the points on the segment 

        if backwards:
            # STEP 2 v2: if it is backwards do  dubins (PATH A)
            # dubins should probably be a case in a star, and just set a variable here to True
            # TODO: implement dubins to do a u turn 
            path_A = None

        # STEP 2: find the closest point on line to start using function above (PATH A)
        # TODO: BEFORE ADDING IN THE INTERPOLATED POINTS IN THE LINE SEGMENT maybe get the slope of the line segment and then find perp line
        # go through each of the starting intersections and find the first two that are closest to the point? then take that line seg
        # then can check if the perp line at that coord goes through point on line. otherwise take the closest intersection line seg

        # p1 is the closest point from dictionary to point
        # p2 is the second closest point theoretically this should not be a problem cuz the line segmenet doesnt have any sharp turns

        a1 = line_closest # this implementation depends on how the line is represented
        a2 = line_behind_of_closest # 
        start_closest = closest_point(a1, a2, self.return_start)
        A_node = AStarNode(self.return_start, start_closest, map)
        path_A, _ = A_node.plan_path()

        # STEP 3: finding the closest point on line to the end (PATH C)
        # STEP 4: get the path from closest point to end point
        # TODO: use same function from first part
        # TODO: see if should change transformation stuff
        c1 = end_line_closest # this implementation depends on how line rep - two closest points to end point
        c2 = end_line_ahead_of_closest
        end_closest = closest_point(c1, c2, self.return_end)
        C_node = AStarNode(self.return_start, end_closest, map)
        path_C, _ = C_node.plan_path()

        # STEP 5: get the indices of the points closest to start and end, then get the segment in between (PATH B)
        path_B = line_traj[near_start : near_end + 1]

        # STEP 6: add all the paths together
        final_path = path_A + path_B + path_C

        self.publish_trajectory(final_path)

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
