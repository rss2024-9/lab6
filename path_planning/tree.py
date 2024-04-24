
import numpy as np
import dubins
import random

turning_rad = 0.848
resolution = 0.0504

class TreeNode:
    def __init__(self, x, y,theta, parent=None,cost=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent  = parent
        self.cost = cost

    def update_cost(self,new_cost):
        self.cost = new_cost

    def update_parent(self,new_parent):
        self.parent = new_parent

def cost_func(p1,p2,turning_radius=turning_rad/resolution):
    """
    Calculates cost for path between two points
    p1 - starting TreeNode
    p2 - ending TreeNode
    """
    
    # # Get the shortest Dubins path
    # path = dubins.shortest_path((p1.x, p1.y, p1.theta), 
    #                             (p2.x, p2.y, p2.theta), 
    #                             turning_radius)
    
    # # Return the length of the path
    # return path.path_length()
    return np.linalg.norm(np.array([p1.x,p1.y]) - np.array([p2.x,p2.y]))



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
    # height, width = map.shape
    # valid_point = False
    # occupied = 0.65 #threshold from map_params
    # free = 0.196
    # # while not valid_point:
    # #     # Generate random x and y coordinates within the map
    # #     x = np.random.randint(0, width)
    # #     y = np.random.randint(0, height)
    # #     if map[y,x] <occupied and map[y,x] > free:
    # #         valid_point =True
    # while not valid_point:
    #     x = np.random.randint(0, width)
    #     y = np.random.randint(0, height)
    #     if map[y,x] ==0:
    #         valid_point = True
    
    # #just a random orientation
    # print(map)
    # print(np.array(map))
    theta = np.random.uniform(0, 2*np.pi)
    point = random.choice(map)

    x=point[1] #column or x value
    y=point[0] #row or y value


    return TreeNode(float(x),float(y),theta)


# def get_nearest_point( tree, point):
#     """
#     Find the nearest point in the tree to the given point
#     params:
#     tree - list of TreeNodes
#     point - TreeNode
#     returns: nearest TreeNode

#     """
    
#     # Extract the coordinates and thetas from the tree nodes and the point
#     tree_coords = np.array([(n.x, n.y) for n in tree])
#     point_coords = np.array([point.x, point.y])
    
#     # Calculate the euclidean distances using vectorized operations
#     distances = np.linalg.norm(tree_coords - point_coords, axis=1)
    
#     # Find the index of the nearest point
#     nearest_index = np.argmin(distances)
    
#     # Return the nearest TreeNode
#     return tree[nearest_index]
    

# def new_state( q_nearest, q_rand, step_size,turning_radius=turning_rad/resolution):
#     """ 
#     Create a new point in the direction of p2 from p1 with distance delta
#     params:
#     q_nearest - TreeNode
#     q_rand - TreeNode
#     step_size - float
#     returns: TreeNode
#     """
#     # distance = euclidean_distance(q_nearest, q_rand)
#     # if distance <= step_size:
#     #     return q_rand
#     # else:
#     #     direction = np.array([q_rand.x,q_rand.y]) - np.array([q_nearest.x,q_nearest.y])
#     #     direction /= distance
#     #     new_point = np.array([q_nearest.x,q_nearest.y]) + direction * step_size
#     #     return TreeNode(new_point[0],new_point[1],np.random.uniform(0,np.pi*2))
    
#     path = dubins.shortest_path((q_nearest.x, q_nearest.y, q_nearest.theta), 
#                                 (q_rand.x, q_rand.y, q_rand.theta), 
#                                 turning_radius)
#     new_points, _ = path.sample_many(step_size)
#     #print(f"x:{q_nearest.x == new_points[0][0]}, y:{q_nearest.y == new_points[0][1]}, theta:{q_nearest.theta == new_points[0][2]}, ")
#     #in case of short path
#     if len(new_points) >=2:
#         point = new_points[1] 
#     else:
#         point = (q_rand.x,q_rand.y,q_rand.theta)
#     return TreeNode(*point)
    

# def is_collision_free( p1, p2, map):
#     # Check if the path between p1 and p2 is free of obstacles

#     # # Convert the poses to grid coordinates
#     # x2, y2 = int(p2.x), int(p2.y)
#     # turning_radius = turning_rad/resolution #turning radius divided by resolution
#     # configs , _ = dubins.shortest_path((p1.x,p1.y,p1.theta),(p2.x,p2.y,p2.theta),turning_radius).sample_many(.25/.0504)
#     # # If the target cell is occupied (probability > 0.65), return True,
#     # configs.append((x2,y2))

#     # if any([map[int(point[1]), int(point[0])] !=0 for point in configs ]):
#     #     return False

#     # # If the target cell is not occupied, return False
#     # return True

#     # Convert the poses to grid coordinates
#     x2, y2 = int(p2.x), int(p2.y)
#     turning_radius = turning_rad / resolution  # turning radius divided by resolution
#     configs, _ = dubins.shortest_path((p1.x, p1.y, p1.theta), (p2.x, p2.y, p2.theta), turning_radius).sample_many(.25 / .0504)
#     # Append target cell to configs
#     configs.append((p2.x, p2.y,p2.theta))

#     # Convert configs to NumPy array for vectorized operations
#     configs_np = np.array(configs, dtype=int)

#     # Check if any of the cells in the path are occupied using vectorized comparison
#     occupied_cells = map[configs_np[:, 1], configs_np[:, 0]] != 0

#     # If any cell in the path is occupied, return False, otherwise return True
#     return not np.any(occupied_cells)
    

# def rewire( tree, q_new, near_nodes, step_size,map):
#     # Rewire the tree to maintain optimality

#     for near_node in near_nodes:
#         if near_node == q_new.parent:
#             continue
#         new_cost = near_node.cost + cost_func(near_node, q_new)
#         if new_cost < q_new.cost and is_collision_free(near_node, q_new, map):
#             q_new.update_parent(near_node)
#             q_new.update_cost(new_cost)
#             update_children_costs(tree, q_new, step_size, map)
    


# def update_children_costs(tree,parent, step_size, map):

#     # for node in parent.children:
#     #     new_cost = parent.cost + euclidean_distance(parent, node)
#     #     if new_cost < node.cost and is_collision_free(parent, node, obstacles):
#     #         node.parent = parent
#     #         node.cost = new_cost
#     #         update_children_costs(node, step_size, obstacles)

#     for node in tree:
#         if node.parent == parent:
#             new_cost = parent.cost + cost_func(parent, node)
#             if new_cost < node.cost and is_collision_free(parent, node, map):
#                 node.parent = parent
#                 node.cost = new_cost
#                 update_children_costs(tree, node, step_size, map)


def create_path( end_point):
    # Create a path from the start to the goal 
    x_old = []
    y_old =[]
    node = end_point
    #get path in reverse order
    while node is not None:
                x_old.append(node.x)
                y_old.append(node.y)
                node = node.parent
    x_old = np.array(x_old)
    y_old = np.array(y_old)
    new_x,new_y = transform_mtw(x_old,y_old)

    path = list(zip(new_x,new_y))
    path.reverse()
    #return path in correct order
    return path
    


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

    return new_x,new_y,new_theta


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





    