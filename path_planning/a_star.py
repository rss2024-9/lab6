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

# why don't you need AStarNode(Node) here
class AStarNode:
    def __init__(self, return_start, return_end, map, pool_size = 5, turning = 0.848, threshold_ang = np.pi):

        # i hope i transfered all the variables that i will need into here but cannot guarantee that i did
        self.map = map

        # CURRENTLY AM NOT USING THIS
        # self.start = start
        # self.end = end

        # transformed start and end that are not rounded for path reconstruction
        self.return_start = return_start
        self.return_end = return_end

        # self.start_theta = None
        # self.end_theta = None

        self.POOL_SIZE = pool_size

        # TODO: implement dubins curves
        self.TURNING_RAD = turning
        self.THRESHOLD_ANGLE = threshold_ang

        # for testing
        self.final_distance = 0

    def plan_path(self):
        """
        returns a path between self.return_start and self.return_end
        """

        start = (int(self.return_start[0]/self.POOL_SIZE), int(self.return_start[1]/self.POOL_SIZE))
        end = (int(self.return_end[0]/self.POOL_SIZE), int(self.return_end[1]/self.POOL_SIZE))

        if not self.is_valid_cell(start) or not self.is_valid_cell(end):
            print("invalid start or end")
            return        

        backwards = False
        visited = set()
        scores = {start: 0}
        previous = {}
        # have f-score and position/node that it corresponds to (start is 0)
        # goes score, position
        queue = [(0, start)]

        while queue:
            current_score, current_node = heapq.heappop(queue)

            # if this is the last node then reconstruct the path
            if current_node == end:
                path = self.reconstruct_path(previous, start, end)

                second_point = transform_wtm(path[2][0], path[2][1], 0)
                if backwards_check(self.return_start[:2], second_point[:2], self.return_start[2]):
                    backwards = True
                    raise NotImplementedError

                return path, backwards

            visited.add(current_node)
            # returns all valid neighbors + visited check is in for loop
            neighbors = self.get_neighbors(current_node)

            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                # f-score is distance to neighbor + neighbor to end
                neighbor_end = manhattan(neighbor, end)
                start_neighbor = current_score + manhattan(current_node, neighbor)
                new_score = start_neighbor + neighbor_end

                # updating distances
                if neighbor not in scores or new_score < scores[neighbor]:
                    scores[neighbor] = new_score
                    previous[neighbor] = (current_node)
                    heapq.heappush(queue, (new_score, neighbor))
        
        print("couldn't find a path")


    # extra functions    
    def get_neighbors(self, cell):
        '''
        takes in point getting the possible neighbors within -1 or 1 of the current point
        '''
        x, y = cell
        neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (((dx != 0) or (dy != 0))) ]
        valid_neighbors = filter(self.is_valid_cell, neighbors)
        
        return valid_neighbors

    def is_valid_cell(self, cell):
        '''
        takes in a point and checks if it is valid in map
        '''
        # x and y are flipped in map
        x = cell[0]
        y = cell[1]
        
        return self.map[int(y), int(x)] == 0

    def reconstruct_path(self, previous, start, end, dubin = False, dubin_path = None):
        '''
        takes in the parent dictionary, start and end point, whether or not to do dubins and return path
        taking the path from the end and then going back up until the start
        '''
        path = [transform_mtw(self.return_end[0], self.return_end[1])]
        current = end

        while current != start :
            current = previous[current]
            path.append(transform_mtw(current[0]*self.POOL_SIZE, current[1]*self.POOL_SIZE))

        path.append(transform_mtw(self.return_start[0], self.return_start[1]))
        path.reverse()

        if dubin:
            path = [transform_mtw(self.return_start[0], self.return_start[1])]
            for point in dubin_path:
                path.append(transform_mtw(point[0]*self.POOL_SIZE, point[1]*self.POOL_SIZE))

        return path


# OUTSIDE FUNCTIONS
def backwards_check(start, second, theta):
    '''
    takes in start point, second point, and starting yaw
    getting the position vector from start to second point and getting the heading vector from start
    dotting the two vectors and seeing if they are negative
    '''
    backwards = False
    pos_array = [second[0] - start[0], second[1] - start[1]]
    head_array = [np.cos(theta)*start[0], np.sin(theta)*start[1]]
    
    pos_vec = np.array(pos_array)
    heading_vec = np.array(head_array)

    dot = heading_vec @ pos_vec

    if dot < 0:
        backwards = True
    
    return backwards

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
    """
    takes in x, y and transforms from map to world scale
    """

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

def distance(cell1, cell2):
    '''
    takes in two points
    calculates the euc distance between two points
    '''
    x1, y1 = cell1
    x2, y2 = cell2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def manhattan(cell1, cell2):
    '''
    takes in two points
    manhattan distance heuristic for a star
    '''
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x2 - x1) + abs(y2 - y1)

def chebyshev(cell1, cell2):
    '''
    takes in two points
    chebyshev distance heuristic for a star
    '''
    x1, y1 = cell1
    x2, y2 = cell2
    return max(abs(x2 - x1), abs(y2 - y1))

def closest_point(p1, p2, point):
    '''
    takes in two points that form a line segment and a point to find the closest point on line 
    p1 should be the closest point on the line segment given to us
    returns the closest point
    '''
    p1 = np.array(p1)
    p2 = np.array(p2)
    point = np.array(point)

    p1_to_p2_vec = p2 - p1
    p1_to_point_vec = point - p1

    # project pi to point onto p1 to p2
    proj_len = np.dot(p1_to_point_vec, p1_to_p2_vec) / np.dot(p1_to_p2_vec, p1_to_p2_vec)
    # if the start or end is behind the closest point then the closest is the closest point
    if proj_len <= 0:
        closest_point = p1
    else:
        closest_point = p1 + proj_len * p1_to_p2_vec

    return closest_point
