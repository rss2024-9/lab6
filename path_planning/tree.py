
import numpy as np


#TODO fix pose stuff, honestly just rewrite the class and functions so all of them just use the Pose class in TreeNode
class TreeNode:
    def __init__(self, x, y, parent=None,cost=0.0):
        self.x = x
        self.y = y
        self.parent  = parent
        self.cost = cost

    def update_cost(self,new_cost):
        self.cost = new_cost

    def update_parent(self,new_parent):
        self.parent = new_parent

def euclidean_distance(p1,p2):
    """
    takes in TreeNodes and returns euclidean distance between them
    params: 
    p1,p2 - TreeNode
    returns:
    float
    """
    return np.linalg.norm(np.array([p1.x,p1.y]) - np.array([p2.x,p2.y]))

def get_random_point(map):
    """
    Generate a random point in the configuration space
    params: 
    map-2D np array
    returns: TreeNode
    """
    height, width = map.shape
    valid_point = False
    occupied = 0.65 #threshold from map_params
    free = 0.196
    # while not valid_point:
    #     # Generate random x and y coordinates within the map
    #     x = np.random.randint(0, width)
    #     y = np.random.randint(0, height)
    #     if map[y,x] <occupied and map[y,x] > free:
    #         valid_point =True
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)

    return TreeNode(float(x),float(y))


def get_nearest_point( tree, point):
    """
    Find the nearest point in the tree to the given point
    params:
    tree - list of TreeNodes
    point - TreeNode
    returns: nearest TreeNode

    """
    
    distances = [euclidean_distance(point, n) for n in tree]
    nearest_index = np.argmin(distances)
    return tree[nearest_index]
    

def new_state( q_nearest, q_rand, step_size):
    """ 
    Create a new point in the direction of p2 from p1 with distance delta
    params:
    q_nearest - TreeNode
    q_rand - TreeNode
    step_size - float
    returns: TreeNode
    """
    distance = euclidean_distance(q_nearest, q_rand)
    if distance <= step_size:
        return q_rand
    else:
        direction = np.array([q_rand.x,q_rand.y]) - np.array([q_nearest.x,q_nearest.y])
        direction /= distance
        new_point = np.array([q_nearest.x,q_nearest.y]) + direction * step_size
        return TreeNode(new_point[0],new_point[1])
    

def is_collision_free( p1, p2, map):
    # Check if the path between p1 and p2 is free of obstacles

    # Convert the poses to grid coordinates
    x2, y2 = int(p2.x), int(p2.y)

    # If the target cell is occupied (probability > 0.65), return True
    if map[y2, x2] !=0:
        return False

    # If the target cell is not occupied, return False
    return True
    

def rewire( tree, q_new, near_nodes, step_size,map):
    # Rewire the tree to maintain optimality

    for near_node in near_nodes:
        if near_node == q_new.parent:
            continue
        new_cost = near_node.cost + euclidean_distance(near_node, q_new)
        if new_cost < q_new.cost and is_collision_free(near_node, q_new, map):
            q_new.update_parent(near_node)
            q_new.update_cost(new_cost)
            update_children_costs(tree, q_new, step_size, map)
    


def update_children_costs(tree,parent, step_size, map):

    # for node in parent.children:
    #     new_cost = parent.cost + euclidean_distance(parent, node)
    #     if new_cost < node.cost and is_collision_free(parent, node, obstacles):
    #         node.parent = parent
    #         node.cost = new_cost
    #         update_children_costs(node, step_size, obstacles)

    for node in tree:
        if node.parent == parent:
            new_cost = parent.cost + euclidean_distance(parent, node)
            if new_cost < node.cost and is_collision_free(parent, node, map):
                node.parent = parent
                node.cost = new_cost
                update_children_costs(tree, node, step_size, map)


def create_path( end_point):
    # Create a path from the start to the goal 
    path = []
    node = end_point
    #get path in reverse order
    while node is not None:
                transform_mtw(node.x, node.y)
                path.append(transform_mtw(node.x, node.y))
                node = node.parent

    #return path in correct order
    return path[::-1]
    


def transform_wtm(x,y):
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

    return new_x,new_y


def transform_mtw(x,y):

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

    return x_new,y_new




from skimage.morphology import erosion, disk

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





    