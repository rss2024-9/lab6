import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
from .tree import *
import numpy as np


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


        self.map_data = None #to be populated with 2D np array
        self.origin = None #Pose msg to draw position and orientation from
        self.start_pose = None

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

        width = msg.info.width
        height = msg.info.height
        self.map_data = np.array(msg.data).reshape(height,width)
        self.origin = self.map.info.origin


        raise NotImplementedError

    def pose_cb(self, pose):
        """
        Sets initial pose for path planning and starts path planning if goal pose and map is set.
        Resets pose after path planning
        """

        self.start_pose = pose.pose
        if self.goal_pose is not None and self.map_data is not None:
            self.plan_path(self.start_pose,self.goal_pose,self.map_data)
            self.start_pose = None
            self.goal_pose = None
        

    def goal_cb(self, msg):
        """
        Sets goal pose for path planning and starts path planning if initial pose and map is set.
        Resets poses after path planning.
        """
        self.goal_pose = msg.pose
        if self.start_pose is not None and self.map_data is not None:
            self.plan_path(self.start_pose,self.goal_pose,self.map_data)
            self.start_pose = None
            self.goal_pose = None
        raise NotImplementedError

    def plan_path(self, start_point, end_point, map, goal_sample_rate = 0.05,step_size=1,max_iter=1000):

        path=False

        #initialize tree with start point
        tree = [Node(start_point)]

        for _ in range(max_iter):

            #sample random point to explore
            q_rand = get_random_point(self.map_data)
            #every once in a while just try to go towards goal point, helps focus search
            if np.random.random() < goal_sample_rate:
                q_rand = Node(end_point)

            #find nearest point in tree to our sampled node
            q_near = get_nearest_point(q_rand, tree)
            new_cost = q_near.cost + euclidean_distance(q_near.position, q_new.position)
            #create new node with nearest node as its parent and input its cost
            q_new = Node(new_state(q_near, q_rand, step_size),q_near,new_cost)
            
            #add new nodes to tree if the move is possible
            if is_collision_free(q_near, q_new, self.map_data):
                near_nodes = [node for node in tree if euclidean_distance(node.position, q_new.position) < rewire_radius]
                tree.append(q_new)
                #update tree to retain optimal paths
                rewire(tree, q_new, near_nodes, step_size, self.map_data)

                #if goal is within reach, just go to goal and return path
                if euclidean_distance(q_new.position, end_point) <= step_size:
                    goal_node = Node(end_point)
                    goal_node.parent = q_new
                    goal_node.cost = q_new.cost + euclidean_distance(q_new.position, goal)
                    tree.append(goal_node)
                    rewire(tree, goal_node, [q_new], step_size, self.map_data)
                    path = True
                    break


        if path:
            for node in tree:
                x,y = node.pose.position.x,node.pose.position.y
                self.trajectory.addPoint((x,y))
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
