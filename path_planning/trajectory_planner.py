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

from .a_star import *


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('map_topic', "/map")
        self.declare_parameter('initial_pose_topic', "/initialpose")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        # self.resolution = 0.5
        # self.step_size = self.resolution
        self.map = None
        self.start_pose = None
        self.end_pose = None

        self.start_theta = None
        self.end_theta = None

        self.return_start = None
        self.return_end = None

        self.RESOLUTION = 0.0504
        self.POOL_SIZE = 5
        self.EROSION_SIZE = 7
        self.DILATION_SIZE = 10
        self.TURNING_RAD = 0.848
        self.THRESHOLD_ANGLE = 3.141

        self.line_traj = {"points": [{"x": -19.99921417236328, "y": 1.3358267545700073}, {"x": -18.433984756469727, "y": 7.590575218200684}, 
                                    {"x": -15.413466453552246, "y": 10.617328643798828}, {"x": -6.186201572418213, "y": 21.114534378051758}, 
                                    {"x": -5.5363922119140625, "y": 25.662315368652344}, {"x": -19.717021942138672, "y": 25.677358627319336}, 
                                    {"x": -20.30797004699707, "y": 26.20694923400879}, {"x": -20.441822052001953, "y": 33.974945068359375}, 
                                    {"x": -55.0716438293457, "y": 34.07769775390625}, {"x": -55.30067825317383, "y": 1.4463690519332886}]}

        self.line_traj_points = []
        self.line_traj_real = []
        for point in self.line_traj["points"]:
            #puts in map frame
            x,y,theta = transform_wtm(point["x"],point["y"],0)

            x/=self.POOL_SIZE
            y/=self.POOL_SIZE

            # do we need to make all of these integers
            x=int(x)
            y=int(y)
            self.line_traj_points.append((x,y,theta))
            self.line_traj_real.append((point["x"],point["y"],0)) #0 in the z part to show its line traj

        self.get_logger().info(f'TRAJ POINTS (transformed and pooled){self.line_traj_points}')
        self.get_logger().info(f'TRAJ REAL{self.line_traj_real}')


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

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        self.get_logger().info("planner initialized")

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
            self.final_path(self.start_pose, self.end_pose, self.map)
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
            self.final_path(self.start_pose, self.end_pose, self.map)
            self.start_pose = None
            self.end_pose = None
            self.get_logger().info("ending goal pose cb")

    def transformations(self, start_pose, end_pose):
        '''
        takes in the start pose and end pose in world frame and then transforms into map
        sets self. start and end theta and self.return start and end to the transformed points
        returns the int of the start and end
        '''
        self.get_logger().info("transforming")

        quaternion = start_pose.pose.pose.orientation
        # Convert the quaternion to Euler angles (roll, pitch, yaw)
        (_, _, yaw) = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.start_theta = yaw

        quaternion = end_pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.end_theta = yaw

        # may have to deal with downsampling here too
        self.return_start = transform_wtm(start_pose.pose.pose.position.x, start_pose.pose.pose.position.y, self.start_theta)
        start = (int(self.return_start[0]/self.POOL_SIZE), int(self.return_start[1]/self.POOL_SIZE))
        self.get_logger().info(f"start point {start[0],start[1]}: {self.map[int(start[1]),int(start[0])]}")
        
        self.return_end = transform_wtm(end_pose.pose.position.x, end_pose.pose.position.y, self.end_theta)
        end = (int(self.return_end[0]/self.POOL_SIZE), int(self.return_end[1]/self.POOL_SIZE))
        self.get_logger().info(f"end point {end[0],end[1]}: {self.map[int(end[1]),int(end[0])]}")

        return start, end

    def pool(self, start, end):
        '''
        takes in self.return_start and self.return_end and returns the x, y coordinates pooled
        '''

        start = (start[0]/self.POOL_SIZE, start[1]/self.POOL_SIZE, start[2])
        end = (end[0]/self.POOL_SIZE, end[1]/self.POOL_SIZE, end[2])
        return start, end

    def valid_trajectory(self, first, second, traj, point, location):
        '''
        takes in the first closest index, the next index, trajectory, and the start or end point, with location as the label
        returns whichever index is in front of the point (where we should start path B trajectory)
        '''
        if first == second:
            return first
        bigger = max(first, second)
        smaller = min(first, second)

        # if it is the start, we want to check if it equals the earlier point
        if location == "start":
            print("point",point,type(point))
            print("traj",traj[smaller],type(traj[smaller]))
            if point == traj[smaller]:
                return smaller
            else:
                return bigger
        elif location == "end":
            if point == traj[bigger]:
                return bigger
            else: 
                return smaller

    def final_path(self, start_pose, end_pose, map):
        '''
        plans a path in three parts:
        A checking backwards going from start point to nearest point on center line
        B taking the trajectory of center line and going until point that is closest to end point on trajectory (probably with lookahead)
        C going from center line to final point

        if backwards:
            A becomes making a u-turn
        '''

        start, end = self.transformations(start_pose, end_pose)

        return_start, return_end = self.pool(self.return_start, self.return_end)

        # POINTS ARE TRANSFORMED INTO CORRECT SCALE AND DOWNSAMPLED PAST HERE

        # STEP 1: checking if the line is backwards
        a_star = AStarNode(return_start, return_end, map)
        opt_path, backwards = a_star.plan_path()
        #print("OPT PATH", opt_path[0])
        #self.publish_trajectory(opt_path)

        # trajectory of the line
        np_points = np.array(self.line_traj_points)[:,0:2] # get x,y for loaded trajectory points

        # TODO: NEED TO KEEP TRACK OF WHICH SIDE WE ARE COMING FROM
        # currently testing with coming from the right
        side = "left"
        if side == "right":
            np_points[::-1]

        # MAKE SURE THAT IF YOU ARE USING CORRECT POOLING SIZE HERE
        np_start = np.array(return_start[:2])
        start_distances = np.linalg.norm(np_points - np_start, axis=1)

        # get closest
        start_args = np.argpartition(start_distances,1)
        start_nearest_ix = start_args[0]
        start_sec_nearest_ix = start_args[1]
        start_next_ix = start_nearest_ix + 1

        # do the same for end
        np_end = np.array(return_end[:2])
        end_distances = np.linalg.norm(np_points - np_end, axis=1)
        end_args = np.argpartition(end_distances, 1)
        self.get_logger().info(f'END ARGS:{end_args}' )
        end_nearest_ix = end_args[0]
        end_prior_ix = end_nearest_ix - 1

        a1 = None
        a2 = None
        a2_ind = None
        path_A = None
        c1 = None
        c2 = None
        c2_ind = None
        path_C = None
        path_B = []
        final_path = []
        start_closest = None
        end_closest = None
        flipped=False

        if backwards:
            self.get_logger().info('GOING BAKCWARDS')
            # STEP 2 v2: if it is backwards do  dubins (PATH A)
            # dubins should probably be a case in a star, and just set a variable here to True
            #             
            offset = 4 # how far away from the line we want the u-turn to go

                    # trajectory of the line
            # np_points = np.array(self.line_traj_points)[:,0:2] # get x,y for loaded trajectory points

            # # idk how much you care about it being ints here, but self.return_start is the decimal version
            # np_start = np.array(start)
            # distances = np.linalg.norm(np_points - np_start, axis=1)

            # # get closest
            # args = np.argpartition(distances,1)
            # nearest_ix = args[0]


            #self.get_logger().info(f"closest points: {nearest_ix,sec_nearest_ix}")

            #check if this is forward vector
            if start_nearest_ix>start_sec_nearest_ix:
                vec = np_points[start_nearest_ix]-np_points[start_sec_nearest_ix]
            else:
                vec = np_points[start_sec_nearest_ix]-np_points[start_nearest_ix]

            a2_ind = len(self.line_traj_real) - start_sec_nearest_ix -1

            normal = vec[::-1]/np.linalg.norm(vec)# get unit normal
            normal[0] = normal[0]*-1*offset #make normal opposite direction and extend it by offset, !TODO idk if this is accurate

            #use projects from function to get nearest point to our position on the trajectory line
            close_point = np.array(closest_point(np_points[start_nearest_ix],np_points[start_sec_nearest_ix],start))
            start_closest = tuple(close_point)

            close_point[0]+=normal[0] # add normal to the close point, !TODO idk if this is correct
            
            #Use dubins to get u-turn trajectory 
            configs, _ = dubins.shortest_path((self.return_start[0]//self.POOL_SIZE, self.return_start[1]//self.POOL_SIZE, self.return_start[2]), (close_point[0], close_point[1], -self.return_start[2]), self.TURNING_RAD/self.POOL_SIZE/self.RESOLUTION).sample_many(.25 / self.RESOLUTION/self.POOL_SIZE)
           
            #transform path to real world
            path_A = [(*transform_mtw(point[0]*self.POOL_SIZE,point[1]*self.POOL_SIZE),1) for point in configs]
            flipped = True
            np_points = np_points[::-1]
            #print("PATH A BAKCWARDS", path_A[0:2])
            
            # self.publish_trajectory(path_A)

        else:
            a1 = np_points[start_nearest_ix]
            
            # if the start index is at the end, take the prior index instead of next
            if start_nearest_ix == len(np_points) - 1:
                self.get_logger().info(f'INDEX: prior {start_nearest_ix - 1}')
                a2 = np_points[start_nearest_ix - 1] # edge case is accounted for in the closest_point function in a_star
                a2_ind = start_nearest_ix - 1
            else:
                 # other wise take the next point
                a2 = np_points[start_next_ix]
                a2_ind = start_next_ix
                self.get_logger().info(f'INDEX: next {start_next_ix}')
            start_closest = closest_point(a1, a2, return_start[:2])
            self.get_logger().info(f'PATH - start closest: {start_closest}, a1: {a1}, a2: {a2}')

            self.get_logger().info(f'PATH - start closest: {start_closest}, return_start: {return_start}')

            A_node = AStarNode(return_start, start_closest, map)
            path_A, _ = A_node.plan_path()

        # STEP 3: finding the closest point on line to the end (PATH C)
        # STEP 4: get the path from closest point to end point
        if flipped:
            end_nearest_ix = len(self.line_traj_real) -end_nearest_ix-1
            end_prior_ix = len(self.line_traj_real) - end_prior_ix -1
            start_nearest_ix = len(self.line_traj_real) -start_nearest_ix -1
            
        c1 = np_points[end_nearest_ix]

        # if the end index is at the start, then take the next index instead of prior
        if end_nearest_ix == 0:
            c2 = np_points[end_nearest_ix + 1]
            c2_ind = end_nearest_ix + 1

        else:
            c2 = np_points[end_prior_ix]
            c2_ind = end_prior_ix

        
        end_closest = closest_point(c1, c2, return_end[:2], angle = return_end[2])
        self.get_logger().info(f'END CLOSEST {end_closest}')
        self.get_logger().info(f'C1 and C2 {end_nearest_ix}, {c2_ind}')
        self.get_logger().info(f'A1 and A2 {start_nearest_ix}, {a2_ind}')
        C_node = AStarNode(end_closest, return_end, map)
        path_C, _ = C_node.plan_path()

        # STEP 5: get the indices of the points closest to start and end, then get the segment in between (PATH B)
        if flipped:
            traj = [row for row in self.line_traj_real[::-1]]
        else:
            traj = [row for row in self.line_traj_real]
        start_index = self.valid_trajectory(start_nearest_ix, a2_ind, traj, start_closest, "start")
        end_index = self.valid_trajectory(end_nearest_ix, c2_ind, traj, end_closest, "end")
        if traj[start_index : end_index] == []:
            final_path = path_A + path_C
        else:
            # path_A -1 should be the closest point on traj to start, path_C 0 should be closest point on traj to end
            # we don't want the ones that are past the places they intersect

            path_B = [path_A[-1]] + traj[start_index : end_index + 1] + [path_C[0]]
            self.get_logger().info(f"THIS IS SPATH B {path_B} [path_A[-1]], {[path_A[-1]]} [path_C[0]] {[path_C[0]]}")

            # STEP 6: add all the paths together
            #final_path = list(path_A[:-1]) + list(path_B) + list(path_C[1:])
            final_path = list(path_A[:-1]) + path_B + list(path_C[1:])
            #print(final_path)

        self.publish_trajectory(final_path)

    def publish_trajectory(self, path):
        if path:
            self.get_logger().info("path planned woohoo")
            for coords in path:
                # made the coords floats cuz it was complaining
                coords = (float(coords[0]), float(coords[1]),float(coords[2]))
                self.trajectory.addPoint(coords)         
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
        else:
            self.get_logger().info("no path, sad")
    

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
