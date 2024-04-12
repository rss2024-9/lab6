
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray,Pose

class TreeNode:
    def __init__(self,pose,parent=None,cost=0.0):
        self.position = pose
        self.parent  = parent
        self.cost = cost

    def update_cost(self,new_cost):
        self.cost = new_cost

    def update_parent(self,new_parent):
        self.parent = new_parent

def euclidean_distance(p1,p2):
    """takes in poses and returns euclidean distance between them"""
    return np.linalg.norm(np.array(p1.position.x,p1.position.y) - np.array(p2.position.x,p2.position.y))

def get_random_point(map):
    """
    Generate a random point in the configuration space
    input: 
    map-2D np array
    output: pose
    """
    height, width = map.shape
    valid_point = False
    occupied = 0.65 #threshold from map_params
    while not valid_point:
        # Generate random x and y coordinates within the map
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        if map[x,y] <occupied:
            valid_point =True


    # Create a Pose message for the random point
    pose = Pose()
    pose.position.x = x
    pose.position.y = y

    return pose


def get_nearest_point(self, tree, point):
    # Find the nearest point in the tree to the given point

    distances = [euclidean_distance(point.pose, n.pose) for n in tree]
    nearest_index = np.argmin(distances)
    return tree[nearest_index]
    

def new_state(self, p1, p2, delta):
    # Create a new point in the direction of p2 from p1 with distance delta
    raise NotImplementedError

def is_collision_free( p1, p2, map):
    # Check if the path between p1 and p2 is free of obstacles
    raise NotImplementedError

def rewire(self, tree, point, radius):
    # Rewire the tree to maintain optimality
    raise NotImplementedError

def create_path(self, tree, start_point, end_point):
    # Create a path from the start to the goal point
    raise NotImplementedError


    