import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray,PointStamped
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

from skimage.morphology import erosion, disk
import numpy as np
import heapq
import pdb
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

        self.map = None
        self.start_pose = None
        self.end_pose = None
        # self.resolution = 0.5
        # self.step_size = self.resolution

        self.return_start = None
        self.return_end = None

        self.test_num = 0
        self.num_tests = 8
        self.run_counter = 0

        self.resolution = 0.0504

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

        self.initial_pub = self.create_publisher(PoseWithCovarianceStamped,
            self.initial_pose_topic,
            10
        )
        self.goal_pub = self.create_publisher(
            PoseStamped,
            "/goal_pose",
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.tester_thing = self.create_subscription(PointStamped,"/clicked_point",self.test_node,10)

    def test_node(self,msg):
        """
        When a point is clicked it starts the next test in the queue
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
        cols = msg.info.width
        rows = msg.info.height
        self.map = erode_map(np.array(msg.data).reshape((rows, cols)), .4/.0504)
        
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
            self.plan_path(self.start_pose, self.end_pose, self.map)
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
            self.plan_path(self.start_pose, self.end_pose, self.map)
            self.start_pose = None
            self.end_pose = None
            self.get_logger().info("ending goal pose cb")

    def plan_path(self, start_point, end_point, map):
        # getting start and end points 
        self.run_counter += 1
        start_time = time.time()
        self.get_logger().info("starting path planning")
        self.return_start = transform_wtm(start_point.pose.pose.position.x, start_point.pose.pose.position.y)
        start = (int(self.return_start[0]), int(self.return_start[1]))
        self.get_logger().info(f"start point {start[0],start[1]}: {self.map[int(start[1]),int(start[0])]}")
        
        self.return_end = transform_wtm(end_point.pose.position.x, end_point.pose.position.y)
        end = (int(self.return_end[0]), int(self.return_end[1]))
        self.get_logger().info(f"end point {end[0],end[1]}: {self.map[int(end[1]),int(end[0])]}")


        if not self.is_valid_cell(start) or not self.is_valid_cell(end):
            self.get_logger().warn("invalid start or goal point")
            return

        visited = set()
        scores = {start: 0}
        previous = {}
        # have f-score and position/node that it corresponds to (start is 0)
        queue = [(0, start)]

        while queue:
            # self.get_logger().info(f'going through queue')
            current_score, current_node = heapq.heappop(queue)

            # if this is the last node then reconstruct the path
            if current_node == end:
                # self.get_logger().info("we have a found a path and are reconstructing")
                path = self.reconstruct_path(previous, start, end)
                end_time = time.time()
                runtime = end_time - start_time
                with open('path_test.txt', 'a') as file:
                    file.write(f'test {self.run_counter}: {runtime} \n')
                    # why is this off???????
                    file.write(f'distance {self.run_counter}: {scores[current_node]*0.0504} \n')
                self.publish_trajectory(path)
                return

            visited.add(current_node)
            # returns all valid neighbors + visited check is in for loop
            neighbors = self.get_neighbors(current_node)

            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                # f-score is distance to neighbor + neighbor to end
                neighbor_end = self.distance(neighbor, end)
                start_neighbor = current_score + self.distance(current_node, neighbor)
                new_score = start_neighbor + neighbor_end

                # updating distances
                if neighbor not in scores or new_score < scores[neighbor]:
                    scores[neighbor] = new_score
                    previous[neighbor] = current_node
                    heapq.heappush(queue, (new_score, neighbor))

                # checking if within a stepsize and moving there (since almost impossible to get exact endpoint)
                # this does not guarantee an optimal path 
                # if neighbor_end <= self.step_size:
                #     scores[end] = float("-inf")
                #     previous[end] = neighbor
                #     heapq.heappush(queue, (scores[end], end))
                #     self.get_logger().info("forcing path to end")
                #     break
        
        self.get_logger().info("outside of while loop")


    # EXTRA FUNCTIONS: 

    def get_neighbors(self, cell):
        '''
        getting the possible neighbors within -1 or 1 of the current
        '''
        x, y = cell
        neighbors = [(x + dx, y + dy) for dx in [-0.5, 0.5] for dy in [-0.5, 0.5]]
        valid_neighbors = filter(self.is_valid_cell, neighbors)
        return valid_neighbors

    def is_valid_cell(self, cell):
        '''
        checking if the cell is a allowed
        '''
        # x and y are flipped
        x = cell[0]
        y = cell[1]
        
        return self.map[int(y), int(x)] == 0

    def distance(self, cell1, cell2):
        '''
        calculates the distance between two points
        '''
        x1, y1 = cell1
        x2, y2 = cell2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def reconstruct_path(self, previous, start, end):
        '''
        taking the path from the end and then going back up
        until the start
        '''
        path = [transform_mtw(self.return_end[0], self.return_end[1])]
        current = end
        while current != start :
            current = previous[current]
            path.append(transform_mtw(current[0], current[1]))
        path.append(transform_mtw(self.return_start[0], self.return_start[1]))
        path.reverse()
        # self.get_logger().info(f"{path}")
        return path

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
    
def erode_map(map_data, erosion_size):
    """
    Erode a binary occupancy grid map.

    Parameters:
    map_data: 2D numpy array representing the map. Obstacles are marked as True.
    erosion_size: The radius of the structuring element used for erosion. Represents the size of the robot.

    Returns:
    Eroded map as a 2D numpy array.
    """

    # Create a disk-shaped structuring element for the erosion
    structuring_element = disk(erosion_size)

    # Erode the map using the structuring element
    eroded_map = erosion(map_data, structuring_element)

    return eroded_map

def transform_mtw(x, y):

    offsets = [25.900000, 48.50000, 3.14]
    x_offset = offsets[0]
    y_offset = offsets[1]
    theta_offset = offsets[2]
    resolution =  0.0504

    #scale
    x*=resolution
    y*=resolution

    #rotate
    c,s = np.cos(theta_offset),np.sin(theta_offset)
    x_temp = c*x -s*y
    y_temp = s*x +c*y
    x_new = x_temp + x_offset
    y_new = y_temp +y_offset

    return x_new, y_new

def transform_wtm(x, y):
    """
    Takes in x,y coordinates from world and transforms them to map scale
    """
    offsets = [25.900000, 48.50000, 3.14]
    x_offset = offsets[0]
    y_offset = offsets[1]
    theta_offset = offsets[2]
    resolution =  0.0504

    #translate and rotate
    temp_x = x-x_offset
    temp_y = y-y_offset
    c,s = np.cos(-theta_offset),np.sin(-theta_offset)
    new_x = c*temp_x - s*temp_y
    new_y = s*temp_x + c*temp_y

    new_x/=resolution
    new_y/=resolution
    
    # changed this to be ints
    return new_x, new_y

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
