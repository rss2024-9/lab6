import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray,PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from .utils import LineTrajectory
from .tree import *
import numpy as np
from tf_transformations import euler_from_quaternion
import time



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
        self.run_counter = 0
        with open('sample_path_test.txt', 'w') as file:
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
            self.initial_pub.publish(start)
            self.goal_pub.publish(end)

        #!TODO continue adding cases
        #write out all the messages with the information they need headers, x,y, and
        #orientation (orientations are really important for RRT so make sure those look reasonable)
        #Use the publish point tool and ros2 topic echo of "/clicked_point" to choose what points to use
        test_conditions = {}

        # short and straight line
        start1 = PoseWithCovarianceStamped()
        end1 = PoseStamped()

        start1.pose.pose.position.x = 24.0  
        start1.pose.pose.position.y = -1.0
        start1.pose.pose.orientation.z = -1.0
        start1.pose.pose.orientation.w = 0.0

        end1.pose.position.x = 12.07  
        end1.pose.position.y = -0.75  
        end1.pose.orientation.z = 1.0
        end1.pose.orientation.w = 0.0

        test_conditions[0] = (start1, end1)

        # backwards goal
        start2 = PoseWithCovarianceStamped()
        end2 = PoseStamped()

        start2.pose.pose.position.x = 12.07
        start2.pose.pose.position.y = -0.75
        start2.pose.pose.orientation.z = 1.0
        start2.pose.pose.orientation.w = 0.0

        end2.pose.position.x = 24.0  
        end2.pose.position.y = -1.0  
        end2.pose.orientation.z = -1.0
        end2.pose.orientation.w = 0.0

        test_conditions[1] = (start2, end2)

        # far away goal through narrow area
        end3 = PoseStamped()

        end3.pose.position.x = -4.5  
        end3.pose.position.y = 23.0  
        end3.pose.orientation.z = 0.5
        end3.pose.orientation.w = 0.85

        test_conditions[2] = (start1, end3)

        # farthest point
        end4 = PoseStamped()

        end4.pose.position.x = -54.5  
        end4.pose.position.y = 35.0  
        end4.pose.orientation.z = 0.7
        end4.pose.orientation.w = 0.72

        test_conditions[3] = (start1, end4)

        # turns
        start3 = PoseWithCovarianceStamped()
        end5 = PoseStamped()

        start3.pose.pose.position.x = -10.0  
        start3.pose.pose.position.y = 17.5
        start3.pose.pose.orientation.z = 0.4
        start3.pose.pose.orientation.w = 0.9

        end5.pose.position.x = -20.0  
        end5.pose.position.y = 29.0  
        end5.pose.orientation.z = 0.75
        end5.pose.orientation.w = 0.68

        test_conditions[4] = (start3, end5)
        conds = test_conditions[self.test_num]
        self.num_tests = len(test_conditions)
        tester(conds[0],conds[1])
        self.test_num+=1
        self.test_num%=self.num_tests



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


        # Convert the eroded map back to a 1D list
        #eroded_map_list = (self.map_data.flatten() * 100).astype(int).tolist()

        # Create a new OccupancyGrid message
        #eroded_map_msg = OccupancyGrid()

        # Set the data field to the eroded map list
        #eroded_map_msg.data = eroded_map_list

        # Set the info field to the original map's info
        #eroded_map_msg.info = msg.info

        #self.map_pub.publish(eroded_map_msg)
        
       

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
        #return
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
    
    def is_collision_free(self, p1, p2,ret_path = False):
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
        if not ret_path:
            # If any cell in the path is occupied, return False, otherwise return True
            return not np.any(occupied_cells)
        else:
            return (not np.any(occupied_cells), configs)
    
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
    
    def new_state(self, q_nearest, q_rand, step_size,turning_radius=turning_rad/resolution):
        """ 
        Create a new point in the direction of p2 from p1 with distance delta
        params:
        q_nearest - TreeNode
        q_rand - TreeNode
        step_size - float
        returns: TreeNode
        """
        # distance = euclidean_distance(q_nearest, q_rand)
        # if distance <= step_size:
        #     return q_rand
        # else:
        #     direction = np.array([q_rand.x,q_rand.y]) - np.array([q_nearest.x,q_nearest.y])
        #     direction /= distance
        #     new_point = np.array([q_nearest.x,q_nearest.y]) + direction * step_size
        #     return TreeNode(new_point[0],new_point[1],np.random.uniform(0,np.pi*2))
        
        path = dubins.shortest_path((q_nearest.x, q_nearest.y, q_nearest.theta), 
                                    (q_rand.x, q_rand.y, q_rand.theta), 
                                    turning_radius)
        new_points, _ = path.sample_many(step_size)
        #print(f"x:{q_nearest.x == new_points[0][0]}, y:{q_nearest.y == new_points[0][1]}, theta:{q_nearest.theta == new_points[0][2]}, ")
        #in case of short path
        ix=1
        point = (q_rand.x,q_rand.y,q_rand.theta)
        points=[]
        
        while ix<len(new_points) and (self.map_data[int(new_points[ix][1]),int(new_points[ix][0])] == 0):
            point = new_points[ix]
            points.append(point)
            ix+=1
        
        return points
    
    def plan_path(self, start_point, end_point, goal_sample_rate = 0.3,step_size=.45/0.0504,max_iter=4000,rewire_radius = 1/0.0504):
        """
        params:
        start_point - PoseWithCovarianceStamped
        end_point - PoseStamped
        map - 2D np array

        """
        self.run_counter += 1
        start_time = time.time()
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

        #check straight line
        has_straight_path, straight_path= self.is_collision_free(self.tree[0],end_point,ret_path=True)
        if has_straight_path:
            for coords in straight_path:
                new_x,new_y = transform_mtw(coords[0],coords[1])
                self.trajectory.addPoint((new_x,new_y))
            end_time = time.time()
            runtime = end_time - start_time
            path_length = 0
            for ix,coords in enumerate(straight_path):
                if ix<len(straight_path)-1:
                    path_length +=np.linalg.norm(np.array([coords[0],coords[1]]) - np.array([straight_path[ix+1][0],straight_path[ix+1][1]]))
            with open('sample_path_test.txt', 'a') as file:
                file.write(f'test {self.run_counter}: {runtime} \n')
                # why is this off???????
                file.write(f'distance {self.run_counter}: {path_length*0.0504} \n')
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
            return


        
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
            # q_new = self.new_state(q_near, q_rand, step_size)
            # new_cost = q_near.cost + cost_func(q_near, q_new)
            # q_new.update_parent(q_near)
            # q_new.update_cost(new_cost)

            q_news = self.new_state(q_near, q_rand, step_size)
            prev = q_near
            for point in q_news:
                
                new_node = TreeNode(*point,parent = prev)
                new_node.update_cost(prev.cost+cost_func(prev,new_node))
                prev = new_node
                self.tree.append(new_node)

            
            #add new nodes to tree if the move is possible
            for indx in range( len(q_news)):
                #self.get_logger().info("in append stage")
                #near_nodes = [node for node in tree if cost_func(node, q_new) < rewire_radius]
                #self.tree.append(q_new)
                #update tree to retain optimal paths
                #rewire(tree, q_new, near_nodes, step_size, self.map_data)
                newest = self.tree[-1-indx]
                #if goal is within reach, just go to goal and return path
                if cost_func(newest, end_point) <= step_size:
                    self.goal_node = end_point
                    goal_node = end_point
                    goal_node.parent = newest
                    goal_node.cost = newest.cost + cost_func(newest, end_point)
                    self.tree.append(goal_node)
                    #rewire(tree, goal_node, [q_new], step_size, self.map_data)
                    path = True
                    break
            if path:
                break


        if path:
            self.get_logger().info(f"path planned woohoo, needed {_} iterations")
            path = create_path(self.goal_node)
            end_time = time.time()
            runtime = end_time - start_time
            with open('sample_path_test.txt', 'a') as file:
                file.write(f'test {self.run_counter}: {runtime} \n')
                # why is this off???????
                file.write(f'distance {self.run_counter}: {end_point.cost*0.0504} \n')
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
