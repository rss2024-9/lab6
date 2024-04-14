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
        self.goal_pose = None
        self.goal_pose = None

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
        self.map_data = erode_map(np.array(msg.data).reshape(height,width),.5/.0504)
        self.origin = msg.info.origin

        

        self.get_logger().info(f"got map, shape: {self.map_data.shape}")
       
        

    def pose_cb(self, pose):
        """
        Sets initial pose for path planning and starts path planning if goal pose and map is set.
        Resets pose after path planning
        params:
        pose - PoseWithCovarianceStamped
        """

        self.start_pose = pose
        self.get_logger().info("got initial pose")
        if self.goal_pose is not None and self.map_data is not None:
            self.trajectory.clear()
            self.plan_path(self.start_pose,self.goal_pose,self.map_data)
            self.start_pose = None
            self.goal_pose = None
            self.goal_node = None
            self.get_logger().info("ending initial pose cb")
        

    def goal_cb(self, msg):
        """
        Sets goal pose for path planning and starts path planning if initial pose and map is set.
        Resets poses after path planning.
        """
        self.goal_pose = msg
        self.get_logger().info("got goal pose")
        if self.start_pose is not None and self.map_data is not None:
            self.trajectory.clear()
            self.plan_path(self.start_pose,self.goal_pose,self.map_data)
            self.start_pose = None
            self.goal_pose = None
            self.goal_node = None
            self.get_logger().info("ending goal pose cb")
        

    def plan_path(self, start_point, end_point, map, goal_sample_rate = 0.05,step_size=1,max_iter=100000,rewire_radius = 2):
        """
        params:
        start_point - PoseWithCovarianceStamped
        end_point - PoseStamped
        map - 2D np array

        """
        self.get_logger().info("starting path planning")
        path=False
        start_point = transform_wtm(start_point.pose.pose.position.x,start_point.pose.pose.position.y)
        self.get_logger().info(f"prob at start point {start_point[0],start_point[1]}: {self.map_data[int(start_point[1]),int(start_point[0])]}")
        end_point = TreeNode(*transform_wtm(end_point.pose.position.x,end_point.pose.position.y))
        #initialize tree with start point
        
        tree = [TreeNode(*start_point)]

        for _ in range(max_iter):

            #sample random point to explore
            q_rand = get_random_point(self.map_data)
            #every once in a while just try to go towards goal point, helps focus search
            if np.random.random() < goal_sample_rate:
                q_rand = end_point

            #find nearest point in tree to our sampled node
            q_near = get_nearest_point(tree,q_rand)
            
            #create new node with nearest node as its parent and input its cost
            q_new = new_state(q_near, q_rand, step_size)
            new_cost = q_near.cost + euclidean_distance(q_near, q_new)
            q_new.update_parent(q_near)
            q_new.update_cost(new_cost)
            
            #add new nodes to tree if the move is possible
            if is_collision_free(q_near, q_new, self.map_data):
                
                near_nodes = [node for node in tree if euclidean_distance(node, q_new) < rewire_radius]
                tree.append(q_new)
                #update tree to retain optimal paths
                rewire(tree, q_new, near_nodes, step_size, self.map_data)

                #if goal is within reach, just go to goal and return path
                if euclidean_distance(q_new, end_point) <= step_size:
                    self.goal_node = end_point
                    goal_node = end_point
                    goal_node.parent = q_new
                    goal_node.cost = q_new.cost + euclidean_distance(q_new, end_point)
                    tree.append(goal_node)
                    rewire(tree, goal_node, [q_new], step_size, self.map_data)
                    path = True
                    break


        if path:
            self.get_logger().info("path planned woohoo")
            path = create_path(self.goal_node)
            for coords in path:
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