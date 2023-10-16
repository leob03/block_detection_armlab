import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import copy
import numpy as np
from scipy.linalg import expm

show_animation = True
verbose = False
D2R = np.pi / 180


## todo: 1. path shortening  done
##       2. path smoothing
##       3. collision for all joints
##       4. knn?

class Node(object):
    """!
    @brief      RRT Node
    @param      joint_angle: 1x5 np.array in radian
    @param      parent: int
    @param      dis: float
    """

    def __init__(self, _joint_angle):
        self.joint_angle = _joint_angle
        self.parent = None
        self.dis = 0


class Obstacle(object):
    """
    (x,y,z): bottom center of cylinder (m) np.array
    r: radius (m)
    h: height (m)
    """

    def __init__(self, _pos, _r, _h):
        self.pos = _pos
        self.r = _r
        self.h = _h


class RRT(object):
    """
    Class for RRT Planning
    """

    def __init__(self, joint_angle_start, joint_angle_end, obstacle_list, test):
        """
        Setting Parameter

        start:Start Position [j1,j2,j3,j4,j5] degree to radian 1x5 np.array
        joint_angle_end:Goal Position [j1,j2,j3,j4,j5] degree to radian
        obstacleList:obstacle Positions [[x,y,z,r,h],...] x,y,z center of cylinder; r radius; h height
        _joint_limit: random sampling Area [[j1_min, j1_max],[j2_min, j2_max]...] 5x2 np.array degree to radian
        """
        self.start = Node(joint_angle_start * D2R)
        self.end = Node(joint_angle_end * D2R)
        # self.joint_limit = _joint_limit * D2R
        self.expandDis = 0.5
        self.goalSampleRate = 0.1
        self.maxIter = 1000
        self.obstacleList = obstacle_list
        self.nodeList = [self.start]
        self.nodeList_b = [self.end]
        self.s_list = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                [0.0, -0.10457, 0.0, -1.0, 0.0, 0.0],
                                [0.0, -0.30457, 0.05, -1.0, 0.0, 0.0],
                                [0.0, -0.30457, 0.25, -1.0, 0.0, 0.0],
                                [-0.30457, 0.0, 0.0, 0.0, 1.0, 0.0]])
        self.m_mat_end = np.array([[0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.42415],
                                   [1.0, 0.0, 0.0, 0.30457],
                                   [0.0, 0.0, 0.0, 1.0]])
        self.m_mat_2 = np.array([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.10391],
                                 [0.0, 0.0, 0.0, 1.0]])
        self.m_mat_3 = np.array([[0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.05],
                                 [1.0, 0.0, 0.0, 0.30391],
                                 [0.0, 0.0, 0.0, 1.0]])
        self.m_mat_4 = np.array([[0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.25],
                                 [1.0, 0.0, 0.0, 0.30391],
                                 [0.0, 0.0, 0.0, 1.0]])

        j1_min = -180
        j1_max = 180
        j2_min = -108
        j2_max = 113
        j3_min = -108
        j3_max = 93
        j4_min = -100
        j4_max = 123
        j5_min = -180
        j5_max = 180
        self.joint_limit = np.array([[j1_min, j1_max],
                                     [j2_min, j2_max],
                                     [j3_min, j3_max],
                                     [j4_min, j4_max],
                                     [j5_min, j5_max]]) * D2R
        # self.path_waitlist = []
        self.test = test
        self.enable_bid = True
            
        # plt.interactive(True)

    #########################
    ## RRT helper function ##
    #########################

    def random_node(self):
        """
        :return:
        """
        joint_angle = np.random.uniform(self.joint_limit[:, 0], self.joint_limit[:, 1])
        node = Node(joint_angle)

        return node

    def joint_norm(self, node1, node2):
        return np.sqrt((node1.joint_angle[0] - node2.joint_angle[0]) ** 2 + 
                       (node1.joint_angle[1] - node2.joint_angle[1]) ** 2 + 
                       (node1.joint_angle[2] - node2.joint_angle[2]) ** 2 + 
                       (node1.joint_angle[3] - node2.joint_angle[3]) ** 2 + 
                       (node1.joint_angle[4] - node2.joint_angle[4]) ** 2)

    def get_nearest_list_index(self, node_list, random_node):
        """
        :param node_list:
        :param rnd:
        :return:
        """
        # d_list = [(node.x - random_node[0]) ** 2 + (node.y - random_node[1]) ** 2 for node in node_list]
        d_list = [self.joint_norm(random_node, node) for node in node_list]
        min_index = d_list.index(min(d_list))
        return min_index

    # def set_nearest_parent(self, new_node):
    def add_and_adjust_tree(self, node_list, new_node: Node, scale=1.2):
        """
        :param new_node:
        :return: parent index
        """
        min_path = (node_list[new_node.parent]).dis
        new_parent = new_node.parent
        # for node in self.nodeList:
        for i, node in enumerate(node_list):
            if 0.01 < self.joint_norm(new_node, node) <= scale * self.expandDis and node.dis < min_path:
                min_path = node.dis
                new_parent = i
        new_node.parent = new_parent
        new_node.dis = node_list[new_parent].dis + self.joint_norm(new_node, node_list[new_parent])
        node_list.append(new_node)
        for i, node in enumerate(node_list):
            if 0.01 < self.joint_norm(new_node,
                                      node) <= scale * self.expandDis and node.dis > new_node.dis + scale * self.expandDis:
                node.parent = len(node_list) - 1
                node.dis = new_node.dis + self.joint_norm(new_node, node)

    def collision_check(self, new_node: Node):
        if len(self.obstacleList) == 0:
            return False  # no collision
        # new_T = self.FK_pox(new_node.joint_angle, self.m_mat_end, self.s_list)
        # if new_T[2,3] < 0:
        #     return 
        links = self.generate_link(new_node)
        for link in links:
            if link[0][2] < 0 or link[1][2] < 0:
                return True
            for obs in self.obstacleList:
                p0 = obs.pos
                p1 = obs.pos + np.array([0, 0, obs.h])
                cyl_obs = [p0, p1, obs.r]
                if self.check_cylinder_collision(link, cyl_obs) == False:
                    return True
        return False

        # path_point = self.FK_pox(new_node.joint_angle, self.m_mat, self.s_list)
        # for obs in self.obstacleList:
        #     pos = obs.pos
        #     dz = pos[2] - path_point[2,3]
        #     if abs(dz) >= obs.h/2:
        #         continue
        #     dis = (pos[0] - path_point[0,3])**2 + (pos[1] - path_point[1,3])**2
        #     if dis <= obs.r**2:
        #         return True # collision
        # return False  # no collision

    def planning(self, joint_angle_start=None, joint_angle_end=None):
        """
        @brief      Path planning main function
        @return     path: list of np.array
        """
        iter = self.maxIter
        if joint_angle_end is not None:
            self.end = Node(joint_angle_end * D2R)
        if joint_angle_start is not None:
            self.start = Node(joint_angle_start * D2R)
        self.nodeList = [self.start]
        self.nodeList_b = [self.end]
        if self.test:
            self.init_plot()

        while iter:
            iter = iter - 1
            # Random Sampling
            if random.random() > self.goalSampleRate:
                rnd = self.random_node()
            else:
                rnd = self.end

            # Find nearest node
            min_index = self.get_nearest_list_index(self.nodeList, rnd)
            # print(min_index)

            # expand tree
            nearest_node = self.nodeList[min_index]

            new_node = copy.deepcopy(nearest_node)
            new_distance = self.joint_norm(rnd, nearest_node)
            new_node.joint_angle += (rnd.joint_angle - nearest_node.joint_angle) / new_distance * self.expandDis
            new_node.parent = min_index

            if self.collision_check(new_node):
                continue

            # new_parent = self.get_nearest_parent(new_node)
            # new_node.parent = new_parent
            # new_node.dis = self.nodeList[new_parent].dis+self.expandDis
            new_node.dis = nearest_node.dis + self.expandDis
            self.nodeList.append(new_node)
            # self.add_and_adjust_tree(self.nodeList, new_node, 1.5)

            self.enable_bid = True
            ############## bi-directional search ##################
            if self.enable_bid:
                min_index_b = self.get_nearest_list_index(self.nodeList_b, new_node)
                nearest_node_b = self.nodeList_b[min_index_b]
                # self.draw_process_temp(new_node, nearest_node_b)  ## for debug
                print("node on 1:", self.node2xyz(new_node))
                print("node on 2:", self.node2xyz(nearest_node_b))
                print("node joint on 1:", new_node.joint_angle)
                print("node joint on 2:", nearest_node_b.joint_angle)
                print("xyz dis:", np.linalg.norm(self.node2xyz(new_node)-self.node2xyz(nearest_node_b)))
                print("joint dis:", self.joint_norm(new_node, nearest_node_b))
                if (self.joint_norm(new_node, nearest_node_b) < self.expandDis):
                    print("Goal!!")
                    self.nodeList_b.append(nearest_node_b)
                    break
                new_node_b = copy.deepcopy(nearest_node_b)
                new_distance_b = self.joint_norm(new_node, nearest_node_b)
                new_node_b.joint_angle += (new_node.joint_angle - nearest_node_b.joint_angle) / new_distance_b * self.expandDis
                new_node_b.parent = min_index_b
                new_node_b.dis = nearest_node_b.dis + self.expandDis

                if self.collision_check(new_node_b):
                    self.draw_process(new_node)
                    continue

                self.nodeList_b.append(new_node_b)
                self.draw_process(new_node, new_node_b)
            else:
                # check goal
                distance = self.joint_norm(new_node, self.end)
                if distance <= self.expandDis:
                    print("Goal!!")
                    break
                self.draw_process(new_node)

        if self.enable_bid:
            return self.generate_path_bid(self.nodeList, self.nodeList_b, self.test)
        else:
            return self.generate_path(self.nodeList, self.test)

    ##########################
    ## plot helper function ##
    ##########################

    # def generate_cylinder(self):
    #     for obs in self.obstacleList:
    #         z = np.linspace(obs.pos[2]-obs.h/2, obs.pos[2]+obs.h/2, 50)
    #         theta = np.linspace(0, 2*np.pi, 50)
    #         theta_grid, z_grid=np.meshgrid(theta, z)
    #         x_grid = obs.r*np.cos(theta_grid) + obs.pos[0]
    #         y_grid = obs.r*np.sin(theta_grid) + obs.pos[1]

    #         self.ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)
    def init_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
        self.ax.set_ylim([0, 0.8])
        self.ax.set_xlim([-0.4, 0.4])
        self.ax.set_zlim([0, 0.8])
        path_point = self.FK_pox(self.start.joint_angle, self.m_mat_end, self.s_list)
        self.ax.scatter(path_point[0, 3], path_point[1, 3], path_point[2, 3], c='orange', marker='o')
        path_point = self.FK_pox(self.end.joint_angle, self.m_mat_end, self.s_list)
        self.ax.scatter(path_point[0, 3], path_point[1, 3], path_point[2, 3], c='g', marker='o')
        for obs in self.obstacleList:
            r = obs.r
            p0 = obs.pos
            p1 = np.array([obs.pos[0], obs.pos[1], obs.pos[2] + obs.h])
            self.generate_cylinder(p0, p1, r)

    def generate_path(self, node_list, xyz):
        # first add end point                 end
        if xyz:
            path_point = self.FK_pox(self.end.joint_angle, self.m_mat_end, self.s_list)
            path = [path_point[:3, 3]]
        else:
            path = [self.end.joint_angle]
        last_index = len(node_list) - 1
        # generate path for tree              end ---> 
        while node_list[last_index].parent is not None:
            node = node_list[last_index]
            if xyz:
                path_point = self.FK_pox(node.joint_angle, self.m_mat_end, self.s_list)
                path.append(path_point[:3, 3])
            else:
                path.append(node.joint_angle)
            last_index = node.parent
        # add start                           end ---> start
        if xyz:
            path_point = self.FK_pox(self.start.joint_angle, self.m_mat_end, self.s_list)
            path.append(path_point[:3, 3])
        else:
            path.append(self.start.joint_angle)
        # reverse                             start ---> end
        path = path[::-1]
        return path

    def generate_path_bid(self, node_list, node_list_b, xyz):
        path = []
        last_index = len(node_list_b) - 1
        # first generate path for tree_b      mid --->
        while node_list_b[last_index].parent is not None:
            node = node_list_b[last_index]
            if xyz:
                path_point = self.FK_pox(node.joint_angle, self.m_mat_end, self.s_list)
                path.append(path_point[:3, 3])
            else:
                path.append(node.joint_angle)
            last_index = node.parent
        # add end                             mid ---> end
        if xyz:
            path_point = self.FK_pox(self.end.joint_angle, self.m_mat_end, self.s_list)
            path.append(path_point[:3, 3])
        else:
            path.append(self.end.joint_angle)
        # reverse                             end ---> mid
        path = path[::-1]
        # generate path for tree              end ---> mid ---> start
        last_index = len(node_list) - 1
        while node_list[last_index].parent is not None:
            node = node_list[last_index]
            if xyz:
                path_point = self.FK_pox(node.joint_angle, self.m_mat_end, self.s_list)
                path.append(path_point[:3, 3])
            else:
                path.append(node.joint_angle)
            last_index = node.parent
        # add start                           end ---> mid ---> start
        if xyz:
            path_point = self.FK_pox(self.start.joint_angle, self.m_mat_end, self.s_list)
            path.append(path_point[:3, 3])
        else:
            path.append(self.start.joint_angle)
        # reverse                             start ---> mid ---> end
        path = path[::-1]
        return path

    def generate_cylinder(self, p0, p1, r):
        # vector in direction of axis
        v = p1 - p0
        # find magnitude of vector
        mag = np.linalg.norm(v)
        # unit vector in direction of axis
        v = v / mag
        # make some vector not in the same direction as v
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])
        # make vector perpendicular to v
        n1 = np.cross(v, not_v)
        # normalize n1
        n1 /= np.linalg.norm(n1)
        # make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)
        # surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        # use meshgrid to make 2d arrays
        t, theta = np.meshgrid(t, theta)
        # generate coordinates for surface
        X, Y, Z = [p0[i] + v[i] * t + r * np.sin(theta) * n1[i] + r * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        self.ax.plot_surface(X, Y, Z)
        # plot axis
        self.ax.plot(*zip(p0, p1), color='red')

    def draw_static(self, path):
        """
        :return:
        """
        # for node in self.nodeList:
        #     if node.parent is not None:
        #         parent_node = self.nodeList[node.parent]
        #         path_point = self.FK_pox(node.joint_angle, self.m_mat_end, self.s_list)
        #         path_point_parent = self.FK_pox(parent_node.joint_angle, self.m_mat_end, self.s_list)
        #         self.ax.plot([path_point[0, 3], path_point_parent[0, 3]], \
        #                      [path_point[1, 3], path_point_parent[1, 3]], \
        #                      [path_point[2, 3], path_point_parent[2, 3]], \
        #                      marker='o', linestyle='-', color='b', markersize=4)
        #         # dis = (path_point[0,3] - path_point_parent[0,3])**2 + \
        #         #       (path_point[1,3] - path_point_parent[1,3])**2 + \
        #         #       (path_point[2,3] - path_point_parent[2,3])**2
        #         # dis = np.sqrt(dis)
        #         # print(dis)
        #         plt.pause(0.01)
        # if self.enable_bid:
        #     for node in self.nodeList_b:
        #         if node.parent is not None:
        #             parent_node = self.nodeList_b[node.parent]
        #             path_point = self.FK_pox(node.joint_angle, self.m_mat_end, self.s_list)
        #             path_point_parent = self.FK_pox(parent_node.joint_angle, self.m_mat_end, self.s_list)
        #             self.ax.plot([path_point[0, 3], path_point_parent[0, 3]], \
        #                          [path_point[1, 3], path_point_parent[1, 3]], \
        #                          [path_point[2, 3], path_point_parent[2, 3]], \
        #                          marker='o', linestyle='-', color='r', markersize=4)
        #             plt.pause(0.01)
        x, y, z = zip(*path)
        self.ax.plot(x, y, z, marker='o', linestyle='-', color='y')

        # Show the plot
        plt.show()

    def draw_process(self, node: Node, node_b: Node = None):
        if node.parent is not None:
            parent_node = self.nodeList[node.parent]
            path_point = self.FK_pox(node.joint_angle, self.m_mat_end, self.s_list)
            path_point_parent = self.FK_pox(parent_node.joint_angle, self.m_mat_end, self.s_list)
            self.ax.plot([path_point[0, 3], path_point_parent[0, 3]], 
                         [path_point[1, 3], path_point_parent[1, 3]], 
                         [path_point[2, 3], path_point_parent[2, 3]], 
                         marker='o', linestyle='-', color='b', markersize=4)
        if node_b is not None and self.enable_bid:
            if node_b.parent is not None:
                parent_node = self.nodeList_b[node_b.parent]
                path_point = self.FK_pox(node_b.joint_angle, self.m_mat_end, self.s_list)
                path_point_parent = self.FK_pox(parent_node.joint_angle, self.m_mat_end, self.s_list)
                self.ax.plot([path_point[0, 3], path_point_parent[0, 3]], 
                             [path_point[1, 3], path_point_parent[1, 3]], 
                             [path_point[2, 3], path_point_parent[2, 3]], 
                             marker='o', linestyle='-', color='r', markersize=4)
        # plt.show(block=True)
        plt.pause(0.1)

    def draw_process_temp(self, node: Node, node_b: Node = None):
        line1 = []
        line2 = []
        if node.parent is not None:
            parent_node = self.nodeList[node.parent]
            path_point = self.FK_pox(node.joint_angle, self.m_mat_end, self.s_list)
            path_point_parent = self.FK_pox(parent_node.joint_angle, self.m_mat_end, self.s_list)
            line1 = self.ax.plot([path_point[0, 3], path_point_parent[0, 3]], 
                         [path_point[1, 3], path_point_parent[1, 3]], 
                         [path_point[2, 3], path_point_parent[2, 3]], 
                         marker='o', linestyle='-', color='b', markersize=4)
            # print(line1)
        if node_b is not None and self.enable_bid:
            if node_b.parent is not None:
                parent_node = self.nodeList_b[node_b.parent]
                path_point = self.FK_pox(node_b.joint_angle, self.m_mat_end, self.s_list)
                path_point_parent = self.FK_pox(parent_node.joint_angle, self.m_mat_end, self.s_list)
                line2 = self.ax.plot([path_point[0, 3], path_point_parent[0, 3]], 
                             [path_point[1, 3], path_point_parent[1, 3]], 
                             [path_point[2, 3], path_point_parent[2, 3]], 
                             marker='o', linestyle='-', color='r', markersize=4)
                # plt.pause(1)
                # line2.remove()
        # plt.show(block=True)
        plt.pause(0.1)
        line1[0].remove()
        if len(line2) > 0:
            line2[0].remove()
        plt.pause(0.1)
    #########################
    ## arm helper function ##
    #########################
    def node2xyz(self, node:Node):
        path_point = self.FK_pox(node.joint_angle, self.m_mat_end, self.s_list)
        return path_point[:3, 3]

    def FK_pox(self, joint_angles, m_mat, s_lst):
        """!
        @brief      Get a  representing the pose of the desired link

        @param      joint_angles  The joint angles np.array in radian
        @param      m_mat         The M matrix
        @param      s_lst         List of screw vectors

        @return     a 4x4 homogeneous matrix representing the pose of the desired link
        """
        H = np.eye(4)
        links = np.shape(s_lst)[0]
        for i in range(links):
            twisted = joint_angles[i] * s_lst[i, :]
            H = H @ expm(self.hat(twisted))
        H = H @ m_mat
        return H

    def hat(self, twist):
        Vx, Vy, Vz, Wx, Wy, Wz = twist
        return np.array([[0, -Wz, Wy, Vx],
                         [Wz, 0, -Wx, Vy],
                         [-Wy, Wx, 0, Vz],
                         [0, 0, 0, 0]])

    def check_cylinder_collision(self, cylinder1, cylinder2):
        ''' 
        Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
        @param     cylinder: List[p0, p1, r]
        @return     True: no collision
        '''
        a0 = cylinder1[0]
        a1 = cylinder1[1]
        r1 = cylinder1[2]
        b0 = cylinder2[0]
        b1 = cylinder2[1]
        r2 = cylinder2[2]
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

        # Calculate denomitator
        A = a1 - a0
        B = b1 - b0
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)
        _A = A / magA
        _B = B / magB
        cross = np.cross(_A, _B)
        denom = np.linalg.norm(cross) ** 2

        # If lines are parallel (denom=0) test if lines overlap.
        # If they don't overlap then there is a closest point solution.
        # If they do overlap, there are infinite closest positions, but there is a closest distance
        if not denom:
            d0 = np.dot(_A, (b0 - a0))

            # Overlap only possible with clamping
            if clampA0 or clampA1 or clampB0 or clampB1:
                d1 = np.dot(_A, (b1 - a0))
                limit = r1 + r2  ## or min(r1, r2)
                # limit = min(r1, r2)

                # Is segment B before A?
                if d0 <= 0 >= d1:
                    if clampA0 and clampB1:
                        if np.absolute(d0) < np.absolute(d1):
                            if verbose:
                                print("result before:", np.linalg.norm(a0 - b0))
                                print("limit:", limit)
                                print(np.linalg.norm(a0 - b0) > limit)
                            return np.linalg.norm(a0 - b0) > limit
                        if verbose:
                            print("result before:", np.linalg.norm(a0 - b1))
                            print("limit:", limit)
                            print(np.linalg.norm(a0 - b1) > limit)
                        return np.linalg.norm(a0 - b1) > limit

                # Is segment B after A?
                elif d0 >= magA <= d1:
                    if clampA1 and clampB0:
                        if np.absolute(d0) < np.absolute(d1):
                            if verbose:
                                print("result after:", np.linalg.norm(a1 - b0))
                                print("limit:", limit)
                                print(np.linalg.norm(a1 - b0) > limit)
                            return np.linalg.norm(a1 - b0) > limit
                        if verbose:
                            print("result after:", np.linalg.norm(a1 - b1))
                            print("limit:", limit)
                            print(np.linalg.norm(a1 - b1) > limit)
                        return np.linalg.norm(a1 - b1) > limit

            # Segments overlap, return distance between parallel segments
            if verbose:
                print("result overlap:", np.linalg.norm(((d0 * _A) + a0) - b0))
                print("limit:", r1 + r2)
                print(np.linalg.norm(((d0 * _A) + a0) - b0) > r1 + r2)
            return np.linalg.norm(((d0 * _A) + a0) - b0) > r1 + r2

        # Lines criss-cross: Calculate the projected closest points
        t = (b0 - a0)
        detA = np.linalg.det([t, _B, cross])
        detB = np.linalg.det([t, _A, cross])
        t0 = detA / denom
        t1 = detB / denom
        pA = a0 + (_A * t0)  # Projected closest point on segment A
        pB = b0 + (_B * t1)  # Projected closest point on segment B

        # Clamp projections
        if clampA0 or clampA1 or clampB0 or clampB1:
            if clampA0 and t0 < 0:
                pA = a0
            elif clampA1 and t0 > magA:
                pA = a1
            if clampB0 and t1 < 0:
                pB = b0
            elif clampB1 and t1 > magB:
                pB = b1
            # Clamp projection A
            if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
                dot = np.dot(_B, (pA - b0))
                if clampB0 and dot < 0:
                    dot = 0
                elif clampB1 and dot > magB:
                    dot = magB
                pB = b0 + (_B * dot)
            # Clamp projection B
            if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
                dot = np.dot(_A, (pB - a0))
                if clampA0 and dot < 0:
                    dot = 0
                elif clampA1 and dot > magA:
                    dot = magA
                pA = a0 + (_A * dot)

        if (np.linalg.norm(pA - a1) == 0 or np.linalg.norm(pA - a0) == 0):
            r1 = r1 / np.sqrt(2)
        if (np.linalg.norm(pB - b1) == 0 or np.linalg.norm(pB - b0) == 0):
            r2 = r2 / np.sqrt(2)
        dis = r1 + r2
        if verbose:
            print("result overlap:", np.linalg.norm(pA - pB))
            print("limit: ", dis)
            print(np.linalg.norm(pA - pB) > dis)
        return np.linalg.norm(pA - pB) > dis

    def inverse_kinematics(self, T):
        l1 = 0.10457
        l2 = np.sqrt(0.2 ** 2 + 0.05 ** 2)
        l3 = 0.2
        l4 = 158.575 / 1000
        offset = np.arctan2(0.05, 0.2)

        p_e = T[0:3, 3]
        R_e = T[0:3, 0:3]
        a_e = R_e[0:3, 2]

        p_w = p_e - l4 * a_e
        # p_w = np.array([-0.125, 0.35, 0.152])
        print(p_w)
        # theta1 = np.arctan2(p_w[1], p_w[0]) ## + np.pi/2
        theta1 = np.arctan2(-p_w[0], p_w[1])
        while (theta1 > np.pi):
            theta1 -= 2 * np.pi
        while (theta1 < -np.pi):
            theta1 += 2 * np.pi

        p_wx = p_w[0]
        p_wy = p_w[1]
        p_wz = p_w[2]
        r_w = np.sqrt(p_wx ** 2 + p_wy ** 2)

        # if p_wz < 20/1000:
        #     p_wz = 20/1000

        theta3 = -np.arccos(
            (p_wx ** 2 + p_wy ** 2 + (p_wz - l1) ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3))  ## or negative
        theta2 = np.arctan2(p_wz - l1, np.sqrt(p_wx ** 2 + p_wy ** 2)) - np.arctan2(l3 * np.sin(theta3),
                                                                                    l2 + l3 * np.cos(theta3))
        # theta2 = np.arctan2((l2 + l3*np.cos(theta3))*p_wz - l3*np.sin(theta3)*r_w, (l2 + l3*np.cos(theta3))*r_w + l3*np.sin(theta3)*p_wz)
        theta3 = -np.pi / 2 + offset - theta3
        # theta2 = t_offset - theta2
        theta2 = np.pi / 2 - offset - theta2
        theta5 = theta1
        theta4 = np.pi / 2 - theta2 - theta3

        return np.array([theta1, theta2, theta3, theta4, theta5])

    def generate_link(self, node: Node):
        '''
        @return     Links: List of Link   List[List[p0, p1, r], ...]
        
        '''
        t2 = self.FK_pox(node.joint_angle, self.m_mat_2, self.s_list[0:2, :])
        link1_start = t2[0:3, 3]
        link1_end = (t2[0:3, 3] + 0.22 * t2[0:3, 2])
        link1 = [link1_start, link1_end, 0.03]

        t3 = self.FK_pox(node.joint_angle, self.m_mat_3, self.s_list[0:3, :])
        link2_start = t3[0:3, 3] - 0.05 * t3[0:3, 2]
        link2_end = t3[0:3, 3] + 0.2 * t3[0:3, 2]
        link2 = [link2_start, link2_end, 0.03]

        t4 = self.FK_pox(node.joint_angle, self.m_mat_4, self.s_list[0:4, :])
        link3_start = t4[0:3, 3] + 0.03 * t4[0:3, 2]
        link3_end = t4[0:3, 3] + 0.13 * t4[0:3, 2]
        link3 = [link3_start, link3_end, 0.03]

        link4_start = t4[0:3, 3] + 0.13 * t4[0:3, 2]
        link4_end = t4[0:3, 3] + 0.17415 * t4[0:3, 2]
        link4 = [link4_start, link4_end, 0.06]
        return [link1, link2, link3, link4]


def main():
    print("start RRT path planning")

    # obstacle_list = [Obstacle((0, 0.3, 0.3), _r=0.05, _h=0.4)]
    # obstacle_list = [Obstacle(np.array([0, 0, -0.2]), _r=2, _h=0.2),
    #                  Obstacle(np.array([0, 0.3, 0]), _r=0.05, _h=0.4)]
    # obstacle_list = [# Obstacle(np.array([0, 0, -0.2]), _r=2, _h=0.2),
    #                  Obstacle(np.array([0.25, 0.075, 0]), _r=0.03, _h=0.4)]
    obstacle_list = [Obstacle(np.array([0.075, -0.05, 0]), _r=0.03, _h=0.16),
                     Obstacle(np.array([-0.075, 0.35, 0]), _r=0.03, _h=0.32),
                     Obstacle(np.array([0, 0.25, 0]), _r=0.03, _h=0.16), ]

    # obstacle_list = []

    # Set Initial parameters
    start = np.array([0, 0, 0, 0, 0])
    # goal = np.array([0,-60,100,90,0])
    # goal = np.array([-1.50748957, -0.31914627,  0.70101839,  1.18892421, -1.50748957])/D2R
    goal = np.array([1.11554625, 0.2174128, 0.04183562, 1.3115479, 1.11554625]) / D2R
    # goal = np.array([89, 0, 0, 0, 0])
    test = True
    rrt = RRT(start, goal, obstacle_list, test)
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
    T[0, 3] = 0.1
    T[1, 3] = 0.25
    T[2, 3] = 0.05
    rrt.end = Node(rrt.inverse_kinematics(T))
    T[0, 3] = -0.2
    T[1, 3] = 0.1
    T[2, 3] = 0.05
    rrt.start = Node(rrt.inverse_kinematics(T))
    # rrt.start = Node(np.array([89, 0, 0, 0, 0])*D2R)
    print(rrt.end.joint_angle)

    # ## link test
    # node = Node(goal*D2R)
    # links = rrt.generate_link(node)
    # for link in links:
    #     # print(link)
    #     rrt.generate_cylinder(link[0], link[1], link[2])
    # plt.show()

    path = rrt.planning()
    # print(path)

    # Draw final path
    if show_animation:
        # plt.close()
        rrt.draw_static(path)


if __name__ == '__main__':
    main()
