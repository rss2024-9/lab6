import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray,PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from .utils import LineTrajectory
from .tree import *
import numpy as np
from tf_transformations import euler_from_quaternion



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
        self.get_logger().info(f"initial pose topic: {self.initial_pose_topic}")
       

        self.map_data = None #to be populated with 2D np array
        self.free_space =None
        self.origin = None #Pose msg to draw position and orientation from
        self.start_pose = None
        self.goal_pose = None
        self.goal_node = None
        self.tree=[]

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

        self.localize_sub = self.create_subscription(Odometry,"/pf/pose/odom",self.localize_cb,10)

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            "/new_map",
            10
        )

        self.test_num = 0
        self.num_tests = 8
        self.click_sub = self.create_subscription(
            PointStamped,
            "/clicked_point",
            self.test_node,
            10
        )
        self.initial_pub = self.create_publisher(PoseWithCovarianceStamped,
            self.initial_pose_topic,
            10
        )
        self.goal_pub = self.create_publisher(
            PoseStamped,
            "/goal_pose",
            10
        )
        

    def localize_cb(self,msg):
        self.start_pose = msg


    def test_node(self,msg):
        """
        When a point is clicked it starts the next test in the queue, karen you can change how this works if you
        want this was just my initial idea of how to test stuff
        """
        def tester(start,end):
            """
            Publish initial positions and goal positions to test path planner
            """
            self.get_logger().info(f"starting test: {self.test_num/self.num_tests}")
            self.initial_pub(start)
            self.goal_pub(end)

        #!TODO continue adding cases
        #write out all the messages with the information they need headers, x,y, and
        #orientation (orientations are really important for RRT so make sure those look reasonable)
        #Use the publish point tool and ros2 topic echo of "/clicked_point" to choose what points to use
        msg1 = PoseWithCovarianceStamped()
        msg2 = PoseStamped()


        test_conditions = {0:(msg1,msg2)}
        conds = test_conditions[self.test_num]
        
        tester(conds[0],conds[1])
        self.test_num+=1%self.num_tests



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
        self.map_data = erode_map(np.array(msg.data).reshape(height,width),.4/.0504)

        zero_indices = np.where(self.map_data == 0)

        # Convert the indices to a list of tuples
        zero_indices_list = list(zip(zero_indices[0], zero_indices[1]))
        self.free_space = zero_indices_list
        
        self.get_logger().info(f"got map, shape: {self.map_data.shape}")
       
        

    def pose_cb(self, pose):
        """
        Sets initial pose for path planning and starts path planning if goal pose and map is set.
        Resets pose after path planning
        params:
        pose - PoseWithCovarianceStamped
        """
        #have return here so it doesn't start doing anything when everything works together
        return
        self.start_pose = pose
        self.get_logger().info("got initial pose")
        if self.goal_pose is not None and self.map_data is not None:
            #clear any old trajectories if they exist and start new path search
            self.trajectory.clear()
            self.plan_path(self.start_pose,self.goal_pose)
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
            #clear any old trajectories if they exist and start new path search
            self.trajectory.clear()
            self.plan_path(self.start_pose,self.goal_pose)
            self.start_pose = None
            self.goal_pose = None
            self.goal_node = None
            self.get_logger().info("ending goal pose cb")

    def get_nearest_point(self,point):
        """
        Find the nearest point in the tree to the given point
        params:
        tree - list of TreeNodes
        point - TreeNode
        returns: nearest TreeNode

        """
        
        # Extract the coordinates and thetas from the tree nodes and the point
        tree_coords = np.array([(n.x, n.y) for n in self.tree])
        point_coords = np.array([point.x, point.y])
        
        # Calculate the euclidean distances using vectorized operations
        distances = np.linalg.norm(tree_coords - point_coords, axis=1)
        
        # Find the index of the nearest point
        nearest_index = np.argmin(distances)
        
        # Return the nearest TreeNode
        return self.tree[nearest_index]
    
    def is_collision_free(self, p1, p2):
        # Check if the path between p1 and p2 is free of obstacles

        # Convert the poses to grid coordinates
        
        turning_radius = turning_rad / resolution  # turning radius divided by resolution
        configs, _ = dubins.shortest_path((p1.x, p1.y, p1.theta), (p2.x, p2.y, p2.theta), turning_radius).sample_many(.25 / .0504)
        # Append target cell to configs
        configs.append((p2.x, p2.y,p2.theta))

        # Convert configs to NumPy array for vectorized operations
        configs_np = np.array(configs, dtype=int)

        # Check if any of the cells in the path are occupied using vectorized comparison
        occupied_cells = self.map_data[configs_np[:, 1], configs_np[:, 0]] != 0

        # If any cell in the path is occupied, return False, otherwise return True
        return not np.any(occupied_cells)
    
    def get_random_point(self):
        """
        Generate a random point in the configuration space
        params: 
        map-2D np array
        returns: TreeNode
        """
        theta = np.random.uniform(0, 2*np.pi)
        point = random.choice(self.free_space)

        x=point[1] #column or x value
        y=point[0] #row or y value


        return TreeNode(float(x),float(y),theta)
    
    def plan_path(self, start_point, end_point, goal_sample_rate = 0.3,step_size=.45/0.0504,max_iter=25000,rewire_radius = 1/0.0504):
        """
        params:
        start_point - PoseWithCovarianceStamped
        end_point - PoseStamped
        map - 2D np array

        """
        self.get_logger().info("starting path planning")
        path=False

        # Get the orientation quaternion from the message
        quaternion = start_point.pose.pose.orientation

        # Convert the quaternion to Euler angles (roll, pitch, yaw)
        (roll, pitch, yaw) = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

        # Get the theta (yaw) component
        theta_start = yaw
        
        # Get the orientation quaternion from the message
        quaternion = end_point.pose.orientation

        # Convert the quaternion to Euler angles (roll, pitch, yaw)
        (roll, pitch, yaw) = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

        # Get the theta (yaw) component
        theta_end = yaw

        start_point = transform_wtm(start_point.pose.pose.position.x,start_point.pose.pose.position.y,theta_start)
        self.get_logger().info(f"prob at start point {start_point[0],start_point[1]}: {self.map_data[int(start_point[1]),int(start_point[0])]}")
        end_point = TreeNode(*transform_wtm(end_point.pose.position.x,end_point.pose.position.y,theta_end))
        #initialize tree with start point
        
        self.tree = [TreeNode(*start_point)]
        goal_offset = .5*step_size
        
        for _ in range(max_iter):
            
            
            #every once in a while just try to go towards goal point, helps focus search
            if np.random.random() < goal_sample_rate:
                # Sample a random point near the goal
                offset_x = np.random.uniform(-goal_offset, goal_offset)
                offset_y = np.random.uniform(-goal_offset, goal_offset)
                q_rand = TreeNode(end_point.x + offset_x, end_point.y + offset_y,end_point.theta)
               
            else:
                #sample random point to explore
                q_rand = self.get_random_point()

            #find nearest point in tree to our sampled node
            q_near = self.get_nearest_point(q_rand)
            
            #create new node with nearest node as its parent and input its cost
            q_new = new_state(q_near, q_rand, step_size)
            new_cost = q_near.cost + cost_func(q_near, q_new)
            q_new.update_parent(q_near)
            q_new.update_cost(new_cost)
            
            #add new nodes to tree if the move is possible
            if self.is_collision_free(q_near, q_new):
                #self.get_logger().info("in append stage")
                #near_nodes = [node for node in tree if cost_func(node, q_new) < rewire_radius]
                self.tree.append(q_new)
                #update tree to retain optimal paths
                #rewire(tree, q_new, near_nodes, step_size, self.map_data)

                #if goal is within reach, just go to goal and return path
                if cost_func(q_new, end_point) <= step_size:
                    self.goal_node = end_point
                    goal_node = end_point
                    goal_node.parent = q_new
                    goal_node.cost = q_new.cost + cost_func(q_new, end_point)
                    self.tree.append(goal_node)
                    #rewire(tree, goal_node, [q_new], step_size, self.map_data)
                    path = True
                    break


        if path:
            self.get_logger().info(f"path planned woohoo, needed {_} iterations")
            path = create_path(self.goal_node)
            for coords in path:
                self.trajectory.addPoint(coords)
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
        else:
            self.get_logger().info("no path, sad")
            
            for point in self.tree:
                new_x,new_y = transform_mtw(point.x, point.y)
                self.trajectory.addPoint((new_x,new_y))
           
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
            
            


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
