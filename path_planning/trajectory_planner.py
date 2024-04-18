import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

import numpy as np
import heapq



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
        self.rows = None
        self.cols = None

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

        self.cols = msg.info.width
        self.rows = msg.info.height
        self.map = np.array(msg.data).reshape((self.rows, self.cols))
        
        # what is origin used for
        self.origin = msg.info.origin

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
        self.get_logger().info(f'{self.end_pose} {self.map}')
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
        self.get_logger().info(f'{self.start_pose} {self.map}')
        if self.start_pose is not None and self.map is not None:
            self.trajectory.clear()
            self.plan_path(self.start_pose, self.end_pose, self.map)
            self.start_pose = None
            self.end_pose = None
            self.get_logger().info("ending goal pose cb")

    def plan_path(self, start_point, end_point, map):
        # getting start and end points 
        self.get_logger().info("starting path planning")
        start = self.transform_wtm(start_point.pose.pose.position.x, start_point.pose.pose.position.y)
        self.get_logger().info(f"start point {start[0],start[1]}: {self.map[int(start[1]),int(start[0])]}")
        
        end = self.transform_wtm(end_point.pose.position.x, end_point.pose.position.y)
        self.get_logger().info(f"end point {end[0],end[1]}: {self.map[int(end[1]),int(end[0])]}")


        if not self.is_valid_cell(start) or not self.is_valid_cell(end):
            self.get_logger().warn("invalid start or goal point")
            return

        # Dijkstra's 
        visited = set()
        scores = {start: 0}
        previous = {}
        # have f-score and position/node that it corresponds to (start is 0)
        queue = [(0, start)]
        # queue = heapq.heapify([(0, start)])

        # im not sure if this is legal if queue is a heap
        while queue:
            self.get_logger().info(f'first element in queue: {queue[0]}')
            current_score, current_node = heapq.heappop(queue)

            # if this is the last node then reconstruct the path
            if current_node == end:
                path = self.reconstruct_path(previous, start, end)
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
                # theoretically distance from current_node to neighbor should be 1
                start_neighbor = current_score + self.distance(current_node, neighbor)
                new_score = start_neighbor + neighbor_end

                # updating distances
                if neighbor not in scores or new_score < scores[neighbor]:
                    scores[neighbor] = new_score

                    # should i be appending to a list here
                    previous[neighbor] = current_node

                    heapq.heappush(queue, (new_score, neighbor))


    # EXTRA FUNCTIONS: 
    def transform_mtw(self, x, y):

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

        return (int(x_new), int(y_new))

    def transform_wtm(self, x, y):
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
        return (int(new_x), int(new_y))

    def get_neighbors(self, cell):
        '''
        getting the possible neighbors within -1 or 1 of the current
        '''
        x, y = cell
        neighbors = [(x + dx, y + dy) for dx in [-1, 1] for dy in [-1, 1]]
        valid_neighbors = filter(self.is_valid_cell, neighbors)
        return valid_neighbors

    def is_valid_cell(self, cell):
        '''
        checking if the cell is a allowed
        '''
        # x and y are flipped
        x = cell[0]
        y = cell[1]
        # self.get_logger().info(f"x : {x}, y : {y}")
        # im not sure if it should be 0 <= seeing if this would help with crashing into walls
        # also not sure what the boundaries are
        # self.get_logger().info(f"check point {self.map[int(cell[1]),int(cell[0])]}")
        # self.get_logger().info(f"cell: {self.map[x,y]}")
        
        return self.map[y, x] == 0

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
        path = [end]
        current = end
        while current != start :
            current = previous[current]
            path.append(self.transform_mtw(current[0], current[1]))
        path.reverse()
        return path

    def publish_trajectory(self, path):
        if path:
            self.get_logger().info("path planned woohoo")
            for coords in path:
                # made the coords floats cuz it was complaining
                self.trajectory.addPoint(float(coords))         
                self.traj_pub.publish(self.trajectory.toPoseArray())
                self.trajectory.publish_viz()
        else:
            self.get_logger().info("no path, sad")
            

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
