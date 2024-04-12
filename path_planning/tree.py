
import numpy as np
from geometry_msgs.msg import PoseWithCovariance, PoseStamped, PoseArray,Pose

class TreeNode:
    def __init__(self, pose, parent=None,cost=0.0):
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
        if map[y,x] <occupied:
            valid_point =True


    # Create a Pose message for the random position
    pose = PoseWithCovariance()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)

    return pose


def get_nearest_point( tree, point):
    # Find the nearest point in the tree to the given point

    distances = [euclidean_distance(point.position.pose, n.position.pose) for n in tree]
    nearest_index = np.argmin(distances)
    return tree[nearest_index]
    

def new_state( q_nearest, q_rand, step_size):
    # Create a new point in the direction of p2 from p1 with distance delta
    distance = euclidean_distance(q_nearest.position.pose, q_rand.position.pose)
    if distance <= step_size:
        return q_rand.position
    else:
        direction = np.array(q_rand.position.pose.position.x,q_rand.position.pose.position.x) - np.array(q_nearest.position.pose.position.x,q_nearest.position.pose.position.y)
        direction /= distance
        new_point = np.array(q_nearest.position.pose.position.x,q_nearest.position.pose.position.y) + direction * step_size
        return tuple(new_point)
    

def is_collision_free( p1, p2, map):
    # Check if the path between p1 and p2 is free of obstacles

    # Convert the poses to grid coordinates
    x2, y2 = int(p2.position.pose.position.x), int(p2.position.pose.position.y)

    # If the target cell is occupied (probability > 0.65), return True
    if map[y2, x2] > 0.65:
        return True

    # If the target cell is not occupied, return False
    return False
    

def rewire( tree, q_new, near_nodes, step_size,map):
    # Rewire the tree to maintain optimality

    for near_node in near_nodes:
        if near_node == q_new.parent:
            continue
        new_cost = near_node.cost + euclidean_distance(near_node.position.pose, q_new.position.pose)
        if new_cost < q_new.cost and is_collision_free(near_node.position.pose, q_new.position.pose, map):
            q_new.update_parent(near_node)
            q_new.update_cost(new_cost)
            update_children_costs(q_new, step_size, map)
    


def update_children_costs(parent, step_size, obstacles):
    for node in parent.children:
        new_cost = parent.cost + euclidean_distance(parent.position.pose, node.position.pose)
        if new_cost < node.cost and is_collision_free(parent, node, obstacles):
            node.parent = parent
            node.cost = new_cost
            update_children_costs(node, step_size, obstacles)


def create_path(self, tree, start_point, end_point):
    # Create a path from the start to the goal point
    raise NotImplementedError


    