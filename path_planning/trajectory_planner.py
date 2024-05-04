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

pi = 3.1415


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
        self.final_distance = 0

        self.start_theta = None
        self.end_theta = None


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
        self.final_distance = 0
        self.run_counter += 1
        start_time = time.time()
        self.get_logger().info("starting path planning")

        quaternion = start_point.pose.pose.orientation
        # Convert the quaternion to Euler angles (roll, pitch, yaw)
        (_, _, yaw) = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.start_theta = yaw

        quaternion = end_point.pose.orientation
        (_, _, yaw) = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.end_theta = yaw

        # may have to deal with downsampling here too
        self.return_start = transform_wtm(start_point.pose.pose.position.x, start_point.pose.pose.position.y, self.start_theta)
        #self.return_start = (self.return_start[0]/self.POOL_SIZE, self.return_start[1]/self.POOL_SIZE)
        start = (int(self.return_start[0]/self.POOL_SIZE), int(self.return_start[1]/self.POOL_SIZE))
        self.get_logger().info(f"start point {start[0],start[1]}: {self.map[int(start[1]),int(start[0])]}")
        
        self.return_end = transform_wtm(end_point.pose.position.x, end_point.pose.position.y, self.end_theta)
        #self.return_end = (self.return_end[0]/self.POOL_SIZE, self.return_end[1]/self.POOL_SIZE)
        end = (int(self.return_end[0]/self.POOL_SIZE), int(self.return_end[1]/self.POOL_SIZE))
        self.get_logger().info(f"end point {end[0],end[1]}: {self.map[int(end[1]),int(end[0])]}")


        if not self.is_valid_cell(start) or not self.is_valid_cell(end):
            self.get_logger().warn("invalid start or goal point")
            return        


        visited = set()
        scores = {start: 0}
        previous = {}
        # have f-score and position/node that it corresponds to (start is 0)
        # goes score, orientation, position
        queue = [(0, start)]

        while queue:
            # self.get_logger().info(f'going through queue')
            current_score, current_node = heapq.heappop(queue)

            # if this is the last node then reconstruct the path
            if current_node == end:
                # self.get_logger().info("we have a found a path and are reconstructing")
                end_time = time.time()

                path = self.reconstruct_path(previous, start, end)
                #if self.backwards_check(start, end):
                second_point = transform_wtm(path[2][0], path[2][1], 0)
                self.get_logger().info(f"BACK start: {self.return_start[:2]}, second: {second_point[:2]}")
                if self.backwards_check(self.return_start[:2], second_point[:2], self.return_start[2]):
                    self.get_logger().info(f"BACKWARDS AHHHHHHHHHHHHHHHHHHH")
                    turning_radius = self.TURNING_RAD / self.RESOLUTION /self.POOL_SIZE # turning radius divided by resolution
                    second_point = transform_wtm(path[2][0], path[2][1], 0)
                    extra_path, _ = dubins.shortest_path((start[0], start[1], self.return_start[2]), (second_point[0]//self.POOL_SIZE, 
                                                        second_point[1]//self.POOL_SIZE, -self.return_start[2]), turning_radius).sample_many(.25 / self.RESOLUTION)
                    extra_path_valid = filter(self.is_valid_cell, extra_path)
                    self.get_logger().info(f"DUBINS DEBUG start:{start}, path: {second_point[0], second_point[1]} theta: {self.start_theta}")
                    self.get_logger().info(f"DUBINS DEBUG: extra_path:{extra_path[0]} {extra_path[-1]}")
                    extra_path = self.reconstruct_path(previous, start, end, dubin=True, dubin_path = extra_path_valid)
                    self.get_logger().info(f"DUBINS DEBUG: extra_path:{extra_path[0]} {extra_path[-1]}")
                    path = extra_path + path[5:]

                self.get_logger().info(f"final_distance: {self.final_distance}")
                runtime = end_time - start_time
                with open('path_test.txt', 'a') as file:
                    file.write(f'test {self.run_counter}: {runtime} \n')
                    file.write(f'distance {self.run_counter}: {self.final_distance} \n')
                    file.write(f'actual distance {self.run_counter}: {self.distance(self.return_start[:2], self.return_end[:2])*self.RESOLUTION} \n')
                self.publish_trajectory(path)
                return

            visited.add(current_node)
            # returns all valid neighbors + visited check is in for loop
            neighbors = self.get_neighbors(current_node)

            for neighbor in neighbors:
                # n_orientation = neighbor[1]
                if neighbor in visited:
                    continue

                # f-score is distance to neighbor + neighbor to end
                #neighbor_end = self.distance(neighbor, end)
                neighbor_end = self.manhattan(neighbor, end)
                start_neighbor = current_score + self.manhattan(current_node, neighbor)
                new_score = start_neighbor + neighbor_end
                # self.get_logger().info(f"new_score: {new_score}")

                # updating distances
                if neighbor not in scores or new_score < scores[neighbor]:
                    scores[neighbor] = new_score
                    # this is unintuitive but the previous dictionary will have the parent node and the orientation from previous node to you
                    previous[neighbor] = (current_node)
                    heapq.heappush(queue, (new_score, neighbor))
        
        self.get_logger().info("outside of while loop")


    # EXTRA FUNCTIONS: 

    def get_neighbors(self, cell):
        '''
        getting the possible neighbors within -1 or 1 of the current
        '''
        # NEED THETA IN HERE AS WELL
        x, y = cell
        # different possible moves and orientation associated with them
        # (dx, dy, theta)
        # states = [(-1, -1, 3*pi/4), (-1, 0, pi/2), (-1, 1, pi/4), 
        #             (0, -1, pi), (0, 1, 0), 
        #             (1, -1, 5*pi/4), (1, 0, 3*pi/2), (1, 1, 7*pi/4)]
        # neighbors = []
        # for state in states:
        #     dx = state[0]
        #     dy = state[1]
        #     orientation = state[2]
        #     if self.is_valid_cell((x + dx, y + dy)):
        #         #self.get_logger().info(f" GET NEIGHBORS: theta: {theta}, orientation: {orientation}, resulting angle: {(theta + orientation)}")
        #         neighbors.append(((x + dx, y + dy), (theta + orientation)%(2*pi)))
        neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (((dx != 0) or (dy != 0))) ]
        #neighbors_x = [(x + dx, y) for dx in [-0.5, 0.5]]
        #neighbors_y = [(x, y + dy) for dy in [-0.5, 0.5]]
        #neighbors = neighbors_x + neighbors_y
        
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
    
    def manhattan(self, cell1, cell2):
        '''
        manhattan distance heuristic for a star
        '''
        x1, y1 = cell1
        x2, y2 = cell2
        return abs(x2 - x1) + abs(y2 - y1)

    def chebyshev(self, cell1, cell2):
        '''
        manhattan distance heuristic for a star
        '''
        x1, y1 = cell1
        x2, y2 = cell2
        return max(abs(x2 - x1), abs(y2 - y1))

    def reconstruct_path(self, previous, start, end, dubin = False, dubin_path = None):
        '''
        taking the path from the end and then going back up
        until the start
        '''
        # backwards = False
        # angles = {transform_mtw(self.return_end[0], self.return_end[1]) : self.return_end[2]}
        path = [transform_mtw(self.return_end[0], self.return_end[1])]
        current = end

        while current != start :
            current = previous[current]
            path.append(transform_mtw(current[0]*self.POOL_SIZE, current[1]*self.POOL_SIZE))
            # angles[transform_mtw(current[0]*self.POOL_SIZE, current[1]*self.POOL_SIZE)] = angle

        path.append(transform_mtw(self.return_start[0], self.return_start[1]))
        # angles[transform_mtw(self.return_start[0], self.return_start[1])] = self.return_start[2]
        path.reverse()

        if dubin:
            path = [transform_mtw(self.return_start[0], self.return_start[1])]
            for point in dubin_path:
                path.append(transform_mtw(point[0]*self.POOL_SIZE, point[1]*self.POOL_SIZE))

        for index in range(0, len(path)-1):
            before = path[index]
            after = path[index + 1]
            self.final_distance += self.distance(before, after)

        # checking if the path is backwards
        # first, _ = transform_wtm(path[0][0], path[0][1], angle[path[0]])
        # second, _ = transform_wtm(path[1][0], path[1][1], angle[path[1]]) 
        # self.get_logger().info(f" BACKWARDS first: {path[0]}, second: {path[1]}, angle 1: {angles[path[0]]}, angle 2: {angles[path[1]]}")
        # self.get_logger().info(f" BACKWARDS first mod: {angles[path[0]]%pi}, secon mod: {angles[path[1]]%pi}, diff: {abs(angles[path[0]]%pi - angles[path[1]]%pi) > pi/2}")

        # if abs(angles[path[0]]%(2*pi) - angles[path[1]]%(2*pi)) > pi/2:
        #     backwards = True

        # second point, relative position from car to second point and dot with heading vector and if negative then the second point is behind the car
        # if you take the arccos of (that)/product of magnitudes ^ then you can get the angle

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
    
    def is_collision_free(self, p1, p2, t1, ret_path = False):
        # Check if the path between p1 and p2 is free of obstacles

        # Convert the poses to grid coordinates
        
        turning_radius = self.TURNING_RAD / self.RESOLUTION  # turning radius divided by resolution
        configs, _ = dubins.shortest_path((p1[0], p1[1], t1), (p2[0], p2[1], -t1), turning_radius).sample_many(.25 / .0504)
        # Append target cell to configs
        configs.append((p2[0], p2[1], -t1))

        # Convert configs to NumPy array for vectorized operations
        configs_np = np.array(configs, dtype=int)

        # Check if any of the cells in the path are occupied using vectorized comparison
        occupied_cells = self.map[configs_np[:, 1], configs_np[:, 0]] != 0
        if not ret_path:
            # If any cell in the path is occupied, return False, otherwise return True
            return not np.any(occupied_cells)
        else:
            return (not np.any(occupied_cells), configs)
    
    def calculate_orientation(self, start_point, end_point):
        # Calculate the differences in x and y coordinates
        delta_x = end_point[0] - start_point[0]
        delta_y = end_point[1] - start_point[1]
        
        # Calculate the angle using arctan2 function
        angle_rad = np.arctan2(delta_y, delta_x)
        
        # Convert the angle from radians to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def backwards_check(self, start, second, theta):
        '''
        takes in start point, second point, and starting yaw
        getting the position vector from start to second point and getting the heading vector from start
        dotting the two vectors and seeing if they are negative
        '''
        backwards = False
        pos_array = [second[0] - start[0], second[1] - start[1]]
        # self.get_logger().info(f"BACK theta: {theta}, return start{self.return_start[2]} x scale: {np.cos(theta)},  y scale: {np.sin(theta)}")
        #head_array = [np.cos(theta)*start[0], np.sin(theta)*start[1]]
        head_array = [np.cos(theta)/self.POOL_SIZE, np.sin(theta)*start[1]]
        
        # self.get_logger().info(f"BACK head array: {head_array},  pos array: {pos_array}")

        pos_vec = np.array(pos_array)
        heading_vec = np.array(head_array)
        # self.get_logger().info(f"BACK head vec: {heading_vec}, pos vec: {pos_vec}")

        dot = heading_vec @ pos_vec

        # mag_head = np.linalg.norm(heading_vec)
        # mag_pos = np.linalg.norm(pos_vec)

        # angle = np.arccos(dot/(mag_head * mag_pos))
        # degrees = np.degrees(angle)
        # self.get_logger().info(f"BACK dot: {dot} heading:{heading_vec}, pos: {pos_vec}, degrees: {degrees}, angle: {angle}")

        # self.get_logger().info(f"BACK mag head: {mag_head}, mag_pos: {mag_pos}, cos term: {dot/(mag_head * mag_pos)}")
        if dot < 0:
            backwards = True
        
        return backwards


        # angle = self.calculate_orientation(start, end)
        # if abs(self.end_theta - angle) > self.THRESHOLD_ANGLE:
        #     self.get_logger().info(f"angle: {angle}, end_theta: {self.end_theta}, diff: {self.end_theta - angle}")
        #     return True
        # if we move in the direction of the orientation and are farther from the end
        # dist = self.distance(start, end)
        # dir_x = abs(np.cos(self.return_start[2]))/self.POOL_SIZE
        # dir_y = abs(np.sin(self.return_start[2]))/self.POOL_SIZE
        # move = (start[0] + dir_x * start[0], start[1] + dir_y * start[1])
        # new_dist = self.distance(move , end)
        # self.get_logger().info(f"BACKWARDS dist:{dist} new_dist:{new_dist}")
        # self.get_logger().info(f"BACKWARDS dir_x:{dir_x} dir_y:{dir_y} start[0]:{start[0]} start[1]:{start[1]}")
        # self.get_logger().info(f"BACKWARDS start:{start} end:{end} move:{move}")
        # # if the distance has increased then we know that the 
        # if dist < new_dist:
        #     return True
        # else:
        #     return False



        
    
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

def dilate_map(map_data, dilation_size):
    """
    dilate a binary occupancy grid map.

    Parameters:
    map_data: 2D numpy array representing the map. Obstacles are marked as True.
    erosion_size: The radius of the structuring element used for dilation. Represents the size of the robot.

    Returns:
    dilated map as a 2D numpy array.
    """

    # Create a disk-shaped structuring element for the erosion
    structuring_element = disk(dilation_size)

    # Erode the map using the structuring element
    dilated_map = dilation(map_data, structuring_element)

    return dilated_map

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

def transform_wtm(x,y,theta):
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

    # adjust theta with the offset and wrap it within the range [0, 2*pi)
    new_theta = np.mod((theta - theta_offset + np.pi), 2 * np.pi) - np.pi

    return new_x, new_y, new_theta


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
