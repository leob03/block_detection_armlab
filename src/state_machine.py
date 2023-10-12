"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
from tkinter import *
import tkinter.messagebox
import cv2
import rrt as motion_planner

D2R = np.pi/180

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """
    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """

        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2, -0.5, -0.3, 0.0, 0.0],
            [0.75*-np.pi/2, 0.5, 0.3, -np.pi/3, np.pi/2],
            [0.5*-np.pi/2, -0.5, -0.3, np.pi / 2, 0.0],
            [0.25*-np.pi/2, 0.5, 0.3, -np.pi/3, np.pi/2],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25*np.pi/2, -0.5, -0.3, 0.0, np.pi/2],
            [0.5*np.pi/2, 0.5, 0.3, -np.pi/3, 0.0],
            [0.75*np.pi/2, -0.5, -0.3, 0.0, np.pi/2],
            [np.pi/2, 0.5, 0.3, -np.pi/3, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]
        
        self.recorded_waypoints = []
        self.holding = False

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == 'record_open':
            self.record_open()

        if self.next_state == 'record_closed':
            self.record_closed()
        
        if self.next_state == 'execute_record':
            self.execute_record()
        
        if self.next_state == 'clear':
            self.clear()

        if self.next_state == 'grab':
            self.grab()

        if self.next_state == 'place':
            self.place()

        if self.next_state == 'calibrate_depth':
            self.calibrate_depth()

        if self.next_state == 'motion_planning':
            self.motion_planning()

        if self.next_state == 'task_1':
            self.task_1()  

        if self.next_state == 'task_2':
            self.task_2()  

        if self.next_state == 'task_3':
            self.task_3() 

        if self.next_state == 'task_4':
            self.task_4() 

    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"
        if self.camera.new_click and self.holding == False:
            self.next_state = "grab"
            self.camera.new_click = False
            print("next state is grab")
        elif self.camera.new_click and self.holding == True:
            self.next_state = "place"
            self.camera.new_click = False

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"
        self.next_state = "idle"
        for points in self.waypoints:
            self.rxarm.set_positions(points)
            time.sleep(2)

    def calibrate_depth(self):
        """!
        @brief      record the value of the depth slope
        """
        self.status_message = "State: Recording depth offset"
        self.current_state = "calibrate_depth"
        self.next_state = "idle"
        if self.camera.depthCalibrated == False:
            self.camera.offset = self.camera.DepthFrameTrans
            self.camera.depthCalibrated == True
            print("depth_calibrated")

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        self.camera.intrinsic_matrix = np.array([[918.3599853515625, 0.0, 661.1923217773438],[0.0,919.1538696289062, 356.59722900390625],[0.0, 0.0, 1.0]], dtype=float)
        # self.camera.intrinsic_matrix = np.array([[898.831565, 0.000000, 709.284902], [0.000000, 897.087012, 402.250269], [0.000000, 0.000000, 1.000000]],dtype=float)
        # self.camera.intrinsic_matrix = np.array([[900, 0.0, 640], [0.0, 900, 360], [0.0, 0.0, 1.0]], dtype=float)
        self.camera.cameraCalibrated = True
        dist_coeffs = np.array([[0.15564486384391785, -0.48568257689476013, -0.0019681642297655344, 0.0007267732871696353, 0.44230175018310547]], dtype=float)
        tag_info = self.camera.tag_detections
        n = len(tag_info.detections)
        image_points = np.zeros((n*5,2), dtype=float)
        for i in range(n):
            image_points[5*i] = [tag_info.detections[i].centre.x,tag_info.detections[i].centre.y]
            image_points[5*i+1] = [tag_info.detections[i].corners[0].x,tag_info.detections[i].corners[0].y]
            image_points[5*i+2] = [tag_info.detections[i].corners[1].x,tag_info.detections[i].corners[1].y]
            image_points[5*i+3] = [tag_info.detections[i].corners[2].x,tag_info.detections[i].corners[2].y]
            image_points[5*i+4] = [tag_info.detections[i].corners[3].x,tag_info.detections[i].corners[3].y]
            
        image_points = np.ascontiguousarray(image_points).reshape((n*5,1,2))
        model_points = self.camera.tag_locations
        (sucess, rot_vec, trans_vec) = cv2.solvePnP(model_points,
                                                    image_points,
                                                    self.camera.intrinsic_matrix,
                                                    dist_coeffs,
                                                    flags=cv2.SOLVEPNP_ITERATIVE)
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        extrinsic_mat = np.eye(4, dtype=float)
        extrinsic_mat[:3,:3] = rot_mat
        extrinsic_mat[:3,3] = trans_vec.flatten()
        self.camera.extrinsic_matrix = extrinsic_mat
        # print(extrinsic_mat)
        self.status_message = "Calibration - Completed Calibration"

        corner_world = np.array([[-500.0,-175.0,0,1],[500.0,-175.0,0,1],[500.0,475.0,0,1],[-500.0,475.0,0,1]], dtype=float).T
        corner_camera = np.matmul(extrinsic_mat,corner_world)
        proj = np.zeros((3,4), dtype=float)
        proj[:3,:3] = np.eye(3, dtype=float)
        corner_pixel = np.matmul(self.camera.intrinsic_matrix, np.matmul(proj, corner_camera))
        for i in range(4):
            corner_pixel[:4,i] = corner_pixel[:4,i]/corner_camera[2,i]
        # src_pts = image_points[:4,].reshape(4,2)
        # dest_pts = np.array([[180.0,540.0],[1040.0,540.0],[1040.0,180.0],[180.0,240.0]])
        src_pts = corner_pixel[:3,].T
        print(src_pts)
        dest_pts = np.array([[140.0,685.0],[1140.0,685.0],[1140.0,35.0],[140.0,35.0]], dtype=float)
        self.camera.Homography = cv2.findHomography(src_pts, dest_pts)[0]


    def grab(self):
        """!
        @brief      Perform the grab of the click and grab
        """
        # print(self.camera.w/1000)
        x,y,z,_ = self.camera.w/1000
        print(type(x),type(y),type(z))        
        print(x,y,z)

        if x == "nan" or y == "nan" or z == "nan":
            print("\n-----------One of the coordinates are NAN-------------\n")
            self.next_state = "idle"
            return

        z = z + 0.1
        T = np.eye(4)
        T[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        # print(T)
        point1, point2 = self.rxarm.get_naive_waypoints(T)


        if point1 == np.array([]) or point2 == np.array([]):
            self.holding = False
        else:
            self.holding = True
            # waypoints, success = self.rxarm.get_naive_waypoints(T)

            # if not success:
            #     self.next_state = "idle"
            #     return
            
            # point1, point2 = waypoints

            # print(point1, point2)
            self.rxarm.gripper.release()
            time.sleep(0.5)
            self.rxarm.set_positions(point1)
            time.sleep(2)
            self.rxarm.set_positions(point2)
            time.sleep(2)
            self.rxarm.gripper.grasp()
            time.sleep(0.5)
            self.rxarm.set_positions(point1)
        self.next_state = "idle"
    
    def place(self):
        self.camera.new_click = False
        # while not self.camera.new_click:
            # print("waiting for click")
        self.camera.new_click = False
        time.sleep(0.5)
        x,y,z,_ = self.camera.w/1000
        z = z + 0.15
        T = np.eye(4)
        T[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        # print(T)
        point1, point2 = self.rxarm.get_naive_waypoints(T)
        # print(point1, point2)
        self.rxarm.set_positions(point1)
        time.sleep(2)
        self.rxarm.set_positions(point2)
        time.sleep(2)
        self.rxarm.gripper.release()
        time.sleep(0.5)
        self.rxarm.set_positions(point1)
        self.holding = False
        self.next_state = "idle"

    def task_1(self):
        self.status_message = "State: Perform Task 1 - Performing Task 1"
        self.current_state = "task_1"
        self.next_state = "idle"
        self.camera.block_position_record = True
        print(self.camera.block_detections)
        time.sleep(2)

        self.rxarm.initialize()
        obstacle_list = [motion_planner.Obstacle(np.array([0.075, -0.05, 0]), _r=0.03, _h=0.16),
                         motion_planner.Obstacle(np.array([0.35, -0.075, 0]), _r=0.03, _h=0.16),]
        # obstacle_list = []

        drop_points_big = np.array([[150.0, -125.0, 0.0],[250.0, -125.0, 0.0],[350.0, -125.0, 0.0], [200.0, -50.0, 0.0],[300.0, -50.0, 0.0]],dtype=float)
        drop_points_small = np.array([[-150.0, -125.0, 0.0],[-250.0, -125.0, 0.0],[-350.0, -125.0, 0.0], [-200.0, -50.0, 0.0],[-300.0, -50.0, 0.0]],dtype=float)

        start = np.array([0,0,0,0,0])
        # goal = np.array([90,0,0,0,0])
        colors = ['red', 'orange', 'blue', 'green', 'yellow', 'purple']
        small = 0
        big = 0
        
        for color in colors:
            if self.camera.block_detections.get(color) is None:
                continue
            else:
                n = len(self.camera.block_detections.get(color))
                for i in range(n):
                    block = self.camera.block_detections[color][i]
                    T = np.eye(4)
                    T[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
                    T[0, 3] = block[0][0]/1000
                    T[1, 3] = block[0][1]/1000
                    T[2, 3] = block[0][2]/1000 + 0.12
                    # point1, point2 = self.rxarm.get_naive_waypoints(T)
                    # point = self.rxarm.get_inverse(block[0][0]/1000, block[0][1]/1000, block[0][2]/1000 + 0.05, 90, block[1])

                    #Go to block
                    point_over_block = self.rxarm.get_inverse(T)
                    # goal = np.array(point)*180/np.pi
                    # rrt = motion_planner.RRT(start, goal, obstacle_list, False)
                    # path = rrt.planning(start, goal)

                    # print(path)
                    # for i in range(len(path)-1,-1,-5):
                    #     self.rxarm.set_positions(path[i].tolist())
                    #     time.sleep(0.5)

                    # self.rxarm.set_positions(path[0].tolist())
                    self.rxarm.set_positions(point_over_block)
                    time.sleep(1.5)

                    #Grab it
                    T1 = np.eye(4)
                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
                    T1[0, 3] = block[0][0]/1000
                    T1[1, 3] = block[0][1]/1000
                    T1[2, 3] = block[0][2]/1000

                    point_block = self.rxarm.get_inverse(T1)
                    # start1 = path[0]
                    # goal1 = np.array(point1)*180/np.pi
                    # # rrt1 = motion_planner.RRT(start1, goal1, obstacle_list, False)
                    # path1 = rrt.planning(start1, goal1)

                    # print(path1[0])
                    # rotated = path1[0]
                    rotated = point_block
                    rotated[4] = rotated[4] - (np.pi/2 - block[1]*np.pi/180)
                    print(rotated)
                    self.rxarm.set_positions(rotated)
                    # self.rxarm.set_positions(path1[0].tolist())
                    time.sleep(2)

                    self.rxarm.gripper.grasp()
                    time.sleep(2)
                    #Go back up
                    self.rxarm.set_positions(point_over_block)
                    time.sleep(2)

                    #Go to Drop Position

                    if block[2] < 1000:
                        drop = drop_points_small[small]
                        small+=1
                    if block[2] >= 1000:
                        drop = drop_points_big[big]
                        big+=1

                    T[0, 3] = drop[0]/1000
                    T[1, 3] = drop[1]/1000
                    T[2, 3] = drop[2]/1000 + 0.04

                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
                    T1[0, 3] = drop[0]/1000
                    T1[1, 3] = drop[1]/1000
                    T1[2, 3] = drop[2]/1000 + 0.1

                    drop_over_position = self.rxarm.get_inverse(T1)
                    # start1 = path[0]
                    # goal1 = np.array(point1)*180/np.pi
                    # # rrt1 = motion_planner.RRT(start1, goal1, obstacle_list, False)
                    # path1 = rrt.planning(start1, goal1)

                    # self.rxarm.set_positions(path1[0].tolist())
                    self.rxarm.set_positions(drop_over_position)
                    time.sleep(2)

                    drop_position = self.rxarm.get_inverse(T)
                    # goal = np.array(point)*180/np.pi
                    # # rrt = motion_planner.RRT(start, goal, obstacle_list, False)
                    # path = rrt.planning(start, goal)

                    self.rxarm.set_positions(drop_position)
                    time.sleep(2)
                    self.rxarm.gripper.release()
                    time.sleep(2)
                    #Drop it

                    # point1 = self.rxarm.get_inverse(T1)
                    # start1 = path[0]
                    # goal1 = np.array(point1)*180/np.pi
                    # # rrt1 = motion_planner.RRT(start1, goal1, obstacle_list, False)
                    # path1 = rrt.planning(start1, goal1)

                    self.rxarm.set_positions(drop_over_position)
                    time.sleep(0.5)

        print("Mission Acomplished !! Wouhou")

    def task_2(self):
        self.status_message = "State: Perform Task 2 - Performing Task 2"
        self.current_state = "task_2"
        self.next_state = "idle"
        self.camera.block_position_record = True
        print(self.camera.block_detections)
        time.sleep(2)

        self.rxarm.initialize()
        obstacle_list = [motion_planner.Obstacle(np.array([0.075, -0.05, 0]), _r=0.03, _h=0.16),
                         motion_planner.Obstacle(np.array([0.35, -0.075, 0]), _r=0.03, _h=0.16),]
        # obstacle_list = []

        # drop_points_big = np.array([[-250.0, -25.0, 0.0],[250.0, -25.0, 0.0],[250.0, 275.0, 0.0], [-250.0, 275.0, 0.0]],dtype=float)
        # drop_points_small = np.array([[-250.0, -25.0, 0.0],[250.0, -25.0, 0.0],[250.0, 275.0, 0.0], [-250.0, 275.0, 0.0]],dtype=float)
        drop_points_big = np.array([[-250.0, -25.0, -5.0],[-250.0, -25.0, 43.0],[-250.0, -25.0, 87.0]],dtype=float)
        drop_points_small = np.array([[250.0, -25.0, -10.0],[250.0, -25.0, 13.0],[250.0, -25.0, 37.0]],dtype=float)

        start = np.array([0,0,0,0,0])
        # goal = np.array([90,0,0,0,0])
        colors = ['red', 'orange', 'blue', 'green', 'yellow', 'purple']
        small = 0
        big = 0
        
        for color in colors:
            if self.camera.block_detections.get(color) is None:
                continue
            else:
                n = len(self.camera.block_detections.get(color))
                for i in range(n):
                    # block = self.camera.block_detections[color][i]
                    # T = np.eye(4)
                    # T[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    # # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
                    # T[0, 3] = block[0][0]/1000
                    # T[1, 3] = block[0][1]/1000
                    # T[2, 3] = block[0][2]/1000 + 0.12
                    # # point1, point2 = self.rxarm.get_naive_waypoints(T)
                    # # point = self.rxarm.get_inverse(block[0][0]/1000, block[0][1]/1000, block[0][2]/1000 + 0.05, 90, block[1])

                    # #Go to block
                    # point = self.rxarm.get_inverse(T)
                    # goal = np.array(point)*180/np.pi
                    # rrt = motion_planner.RRT(start, goal, obstacle_list, False)
                    # path = rrt.planning(start, goal)
                    # # print(path)
                    # # for i in range(len(path)-1,-1,-5):
                    # #     self.rxarm.set_positions(path[i].tolist())
                    # #     time.sleep(0.5)
                    # self.rxarm.set_positions(path[0].tolist())
                    # # print(path[0])
                    # # rotated = path[0]
                    # # rotated[4] = rotated[4] - block[1]
                    # # print(rotated)
                    # # time.sleep(2)
                    # # self.rxarm.set_positions(rotated.tolist())
                    # time.sleep(2)

                    # #Grab it
                    # T1 = np.eye(4)
                    # T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    # # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
                    # T1[0, 3] = block[0][0]/1000
                    # T1[1, 3] = block[0][1]/1000
                    # T1[2, 3] = block[0][2]/1000

                    # point1 = self.rxarm.get_inverse(T1)
                    # start1 = path[0]
                    # goal1 = np.array(point1)*180/np.pi
                    # # rrt1 = motion_planner.RRT(start1, goal1, obstacle_list, False)
                    # path1 = rrt.planning(start1, goal1)
                    # print(path1[0])
                    # rotated = path1[0]
                    # rotated[4] = rotated[4] - (np.pi/2 - block[1]*np.pi/180)
                    # print(rotated)
                    # self.rxarm.set_positions(rotated.tolist())
                    # # self.rxarm.set_positions(path1[0].tolist())
                    # time.sleep(2)

                    # self.rxarm.gripper.grasp()
                    # time.sleep(1)

                    # #Go back up
                    # self.rxarm.set_positions(path[0].tolist())

                    # #Go to Drop Position

                    # if block[2] < 1000:
                    #     drop = drop_points_small[small]
                    #     small+=1
                    # if block[2] >= 1000:
                    #     drop = drop_points_big[big]
                    #     big+=1

                    # T[0, 3] = drop[0]/1000
                    # T[1, 3] = drop[1]/1000
                    # T[2, 3] = drop[2]/1000 + 0.04

                    # T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    # # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
                    # T1[0, 3] = drop[0]/1000
                    # T1[1, 3] = drop[1]/1000
                    # T1[2, 3] = drop[2]/1000 + 0.1

                    # point1 = self.rxarm.get_inverse(T1)
                    # start1 = path[0]
                    # goal1 = np.array(point1)*180/np.pi
                    # # rrt1 = motion_planner.RRT(start1, goal1, obstacle_list, False)
                    # path1 = rrt.planning(start1, goal1)

                    # self.rxarm.set_positions(path1[0].tolist())
                    # time.sleep(2)

                    # point = self.rxarm.get_inverse(T)
                    # goal = np.array(point)*180/np.pi
                    # # rrt = motion_planner.RRT(start, goal, obstacle_list, False)
                    # path = rrt.planning(start, goal)

                    # self.rxarm.set_positions(path[0].tolist())
                    # time.sleep(3)
                    # self.rxarm.gripper.release()

                    # #Drop it

                    # self.rxarm.set_positions(path1[0].tolist())
                    # time.sleep(1)
                    block = self.camera.block_detections[color][i]
                    T = np.eye(4)
                    T[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
                    T[0, 3] = block[0][0]/1000
                    T[1, 3] = block[0][1]/1000
                    T[2, 3] = block[0][2]/1000 + 0.12
                    # point1, point2 = self.rxarm.get_naive_waypoints(T)
                    # point = self.rxarm.get_inverse(block[0][0]/1000, block[0][1]/1000, block[0][2]/1000 + 0.05, 90, block[1])

                    #Go to block
                    point_over_block = self.rxarm.get_inverse(T)
                    # goal = np.array(point)*180/np.pi
                    # rrt = motion_planner.RRT(start, goal, obstacle_list, False)
                    # path = rrt.planning(start, goal)

                    # print(path)
                    # for i in range(len(path)-1,-1,-5):
                    #     self.rxarm.set_positions(path[i].tolist())
                    #     time.sleep(0.5)

                    # self.rxarm.set_positions(path[0].tolist())
                    self.rxarm.set_positions(point_over_block)
                    time.sleep(1.5)

                    #Grab it
                    T1 = np.eye(4)
                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
                    T1[0, 3] = block[0][0]/1000
                    T1[1, 3] = block[0][1]/1000
                    T1[2, 3] = block[0][2]/1000

                    point_block = self.rxarm.get_inverse(T1)
                    # start1 = path[0]
                    # goal1 = np.array(point1)*180/np.pi
                    # # rrt1 = motion_planner.RRT(start1, goal1, obstacle_list, False)
                    # path1 = rrt.planning(start1, goal1)

                    # print(path1[0])
                    # rotated = path1[0]
                    rotated = point_block
                    rotated[4] = rotated[4] - (np.pi/2 - block[1]*np.pi/180)
                    print(rotated)
                    self.rxarm.set_positions(rotated)
                    # self.rxarm.set_positions(path1[0].tolist())
                    time.sleep(2)

                    self.rxarm.gripper.grasp()
                    time.sleep(2)
                    #Go back up
                    self.rxarm.set_positions(point_over_block)
                    time.sleep(2)

                    #Go to Drop Position

                    if block[2] < 1000:
                        drop = drop_points_small[small]
                        small+=1
                    if block[2] >= 1000:
                        drop = drop_points_big[big]
                        big+=1

                    T[0, 3] = drop[0]/1000
                    T[1, 3] = drop[1]/1000
                    T[2, 3] = drop[2]/1000 + 0.04

                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    # T[:3, 3] = np.array([[x], [y], [z]]).reshape(3, 1)
                    T1[0, 3] = drop[0]/1000
                    T1[1, 3] = drop[1]/1000
                    T1[2, 3] = drop[2]/1000 + 0.1

                    drop_over_position = self.rxarm.get_inverse(T1)
                    # start1 = path[0]
                    # goal1 = np.array(point1)*180/np.pi
                    # # rrt1 = motion_planner.RRT(start1, goal1, obstacle_list, False)
                    # path1 = rrt.planning(start1, goal1)

                    # self.rxarm.set_positions(path1[0].tolist())
                    self.rxarm.set_positions(drop_over_position)
                    time.sleep(2)

                    drop_position = self.rxarm.get_inverse(T)
                    # goal = np.array(point)*180/np.pi
                    # # rrt = motion_planner.RRT(start, goal, obstacle_list, False)
                    # path = rrt.planning(start, goal)

                    self.rxarm.set_positions(drop_position)
                    time.sleep(2)
                    self.rxarm.gripper.release()
                    time.sleep(2)
                    #Drop it

                    # point1 = self.rxarm.get_inverse(T1)
                    # start1 = path[0]
                    # goal1 = np.array(point1)*180/np.pi
                    # # rrt1 = motion_planner.RRT(start1, goal1, obstacle_list, False)
                    # path1 = rrt.planning(start1, goal1)

                    self.rxarm.set_positions(drop_over_position)
                    time.sleep(0.5)
        print("Mission Acomplished !! Wouhou")

    def task_3(self):
        self.status_message = "State: Perform Task 3 - Performing Task 3"
        self.current_state = "task_3"
        self.next_state = "idle"
        self.camera.block_position_record = True
        print(self.camera.block_detections)
        time.sleep(2)

        self.rxarm.initialize()

        obstacle_list = []

        drop_points = np.array([[-250.0, 25.0, -10.0],[250.0, 25.0, -10.0],[-250.0, 125.0, -5.0],[250.0, 125.0, -10.0],[-250.0, 225.0, -7.0],[250.0, 225.0, 0.0]],dtype=float)

        start = np.array([0,0,0,0,0])
        # goal = np.array([90,0,0,0,0])
        colors = ['red', 'orange', 'blue', 'green', 'yellow', 'purple']
        bl_nb = 0
        
        for color in colors:
            if self.camera.block_detections.get(color) is None:
                continue
            else:
                n = len(self.camera.block_detections.get(color))
                for i in range(n):
                    block = self.camera.block_detections[color][i]
                    T = np.eye(4)
                    T[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T[0, 3] = block[0][0]/1000
                    T[1, 3] = block[0][1]/1000
                    T[2, 3] = block[0][2]/1000 + 0.12


                    #Go to block
                    point_over_block = self.rxarm.get_inverse(T)
                    self.rxarm.set_positions(point_over_block)
                    time.sleep(1.5)

                    #Grab it
                    T1 = np.eye(4)
                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T1[0, 3] = block[0][0]/1000
                    T1[1, 3] = block[0][1]/1000
                    T1[2, 3] = block[0][2]/1000

                    point_block = self.rxarm.get_inverse(T1)

                    rotated = point_block
                    rotated[4] = rotated[4] - (np.pi/2 - block[1]*np.pi/180)
                    self.rxarm.set_positions(rotated)
                    time.sleep(2)

                    self.rxarm.gripper.grasp()
                    time.sleep(1)
                    
                    #Go back up
                    self.rxarm.set_positions(point_over_block)
                    
                    time.sleep(2)

                    #Go to Drop Position
                    drop = drop_points[bl_nb]
                    bl_nb+=1

                    T[0, 3] = drop[0]/1000
                    T[1, 3] = drop[1]/1000
                    T[2, 3] = drop[2]/1000 + 0.04

                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T1[0, 3] = drop[0]/1000
                    T1[1, 3] = drop[1]/1000
                    T1[2, 3] = drop[2]/1000 + 0.1

                    drop_over_position = self.rxarm.get_inverse(T1)
                    self.rxarm.set_positions(drop_over_position)
                    time.sleep(2)

                    drop_position = self.rxarm.get_inverse(T)
                    self.rxarm.set_positions(drop_position)
                    time.sleep(2)
                    self.rxarm.gripper.release()
                    time.sleep(2)

                    #Go back up

                    self.rxarm.set_positions(drop_over_position)
                    time.sleep(2)

        print("space cleared")

        self.rxarm.sleep()

        time.sleep(2)

        self.camera.block_position_record = True
        print(self.camera.block_detections)
        time.sleep(2)

        self.rxarm.initialize()

        drop_points_colors = np.array([[-125.0, 200.0, -5.0],[-80.0, 200.0, -5.0],[-35.0, 200.0, -5.0],[10.0, 200.0, -5.0],[55.0, 200.0, -5.0],[100.0, 200.0, -5.0]],dtype=float)

        start = np.array([0,0,0,0,0])
        # goal = np.array([90,0,0,0,0])
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        bl_nb = 0
        
        for color in colors:
            if self.camera.block_detections.get(color) is None:
                continue
            else:
                print(self.camera.block_detections)
                n = len(self.camera.block_detections.get(color))
                for i in range(n):
                    block = self.camera.block_detections[color][i]
                    T = np.eye(4)

                    T[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T[0, 3] = block[0][0]/1000
                    T[1, 3] = block[0][1]/1000
                    T[2, 3] = block[0][2]/1000 + 0.12


                    #Go to block
                    point_over_block = self.rxarm.get_inverse(T)
                    self.rxarm.set_positions(point_over_block)
                    time.sleep(1.5)

                    #Grab it
                    T1 = np.eye(4)
                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T1[0, 3] = block[0][0]/1000
                    T1[1, 3] = block[0][1]/1000
                    T1[2, 3] = block[0][2]/1000

                    point_block = self.rxarm.get_inverse(T1)

                    rotated = point_block
                    rotated[4] = rotated[4] - (np.pi/2 - block[1]*np.pi/180)
                    self.rxarm.set_positions(rotated)
                    time.sleep(2)

                    self.rxarm.gripper.grasp()
                    time.sleep(3)
                    
                    #Go back up
                    self.rxarm.set_positions(point_over_block)
                    time.sleep(2)

                    #Go to Drop Position
                    drop_colour = drop_points_colors[bl_nb]
                    bl_nb+=1

                    T[0, 3] = drop_colour[0]/1000
                    T[1, 3] = drop_colour[1]/1000
                    T[2, 3] = drop_colour[2]/1000 + 0.04

                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T1[0, 3] = drop_colour[0]/1000
                    T1[1, 3] = drop_colour[1]/1000
                    T1[2, 3] = drop_colour[2]/1000 + 0.1

                    drop_over_position = self.rxarm.get_inverse(T1)

                    rotated_drop_over = drop_over_position
                    rotated_drop_over[4] = rotated_drop_over[4] - (np.pi/2)
                    self.rxarm.set_positions(rotated_drop_over)
                    time.sleep(2)

                    # self.rxarm.set_positions(drop_over_position)
                    # time.sleep(2)

                    drop_position = self.rxarm.get_inverse(T)
                    rotated_drop = drop_position
                    rotated_drop[4] = rotated_drop[4] - (np.pi/2)
                    self.rxarm.set_positions(rotated_drop)
                    time.sleep(2)


                    # self.rxarm.set_positions(drop_position)
                    time.sleep(2)
                    self.rxarm.gripper.release()
                    time.sleep(2)

                    #Go back up

                    self.rxarm.set_positions(rotated_drop_over)
                    time.sleep(2)

    def task_4(self):
        self.status_message = "State: Perform Task 4 - Performing Task 4"
        self.current_state = "task_4"
        self.next_state = "idle"
        self.camera.block_position_record = True
        print(self.camera.block_detections)
        time.sleep(2)

        self.rxarm.initialize()

        obstacle_list = []

        drop_points = np.array([[-250.0, 25.0, -10.0],[250.0, 25.0, -10.0],[-250.0, 125.0, -5.0],[250.0, 125.0, -10.0],[-250.0, 225.0, -7.0],[250.0, 225.0, 0.0]],dtype=float)

        start = np.array([0,0,0,0,0])
        # goal = np.array([90,0,0,0,0])
        colors = ['red', 'orange', 'blue', 'green', 'yellow', 'purple']
        bl_nb = 0
        
        for color in colors:
            if self.camera.block_detections.get(color) is None:
                continue
            else:
                n = len(self.camera.block_detections.get(color))
                for i in range(n):
                    block = self.camera.block_detections[color][i]
                    T = np.eye(4)
                    T[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T[0, 3] = block[0][0]/1000
                    T[1, 3] = block[0][1]/1000
                    T[2, 3] = block[0][2]/1000 + 0.12


                    #Go to block
                    point_over_block = self.rxarm.get_inverse(T)
                    self.rxarm.set_positions(point_over_block)
                    time.sleep(1.5)

                    #Grab it
                    T1 = np.eye(4)
                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T1[0, 3] = block[0][0]/1000
                    T1[1, 3] = block[0][1]/1000
                    T1[2, 3] = block[0][2]/1000

                    point_block = self.rxarm.get_inverse(T1)

                    rotated = point_block
                    rotated[4] = rotated[4] - (np.pi/2 - block[1]*np.pi/180)
                    self.rxarm.set_positions(rotated)
                    time.sleep(2)

                    self.rxarm.gripper.grasp()
                    time.sleep(1)
                    
                    #Go back up
                    self.rxarm.set_positions(point_over_block)
                    
                    time.sleep(2)

                    #Go to Drop Position
                    drop = drop_points[bl_nb]
                    bl_nb+=1

                    T[0, 3] = drop[0]/1000
                    T[1, 3] = drop[1]/1000
                    T[2, 3] = drop[2]/1000 + 0.04

                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T1[0, 3] = drop[0]/1000
                    T1[1, 3] = drop[1]/1000
                    T1[2, 3] = drop[2]/1000 + 0.1

                    drop_over_position = self.rxarm.get_inverse(T1)
                    self.rxarm.set_positions(drop_over_position)
                    time.sleep(2)

                    drop_position = self.rxarm.get_inverse(T)
                    self.rxarm.set_positions(drop_position)
                    time.sleep(2)
                    self.rxarm.gripper.release()
                    time.sleep(2)

                    #Go back up

                    self.rxarm.set_positions(drop_over_position)
                    time.sleep(2)

        print("space cleared")

        self.rxarm.sleep()

        time.sleep(2)

        self.camera.block_position_record = True
        print(self.camera.block_detections)
        time.sleep(2)

        self.rxarm.initialize()

        drop_points_colors = np.array([[0.0,175.9,-14.0],[0.0,175.9,10.0],[0.0,175.9,34.0],[0.0,175.9,61.0],[0.0,175.9,86.0],[0.0,175.9,111.0]],dtype=float)

        start = np.array([0,0,0,0,0])
        # goal = np.array([90,0,0,0,0])
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        bl_nb = 0
        
        for color in colors:
            if self.camera.block_detections.get(color) is None:
                continue
            else:
                print(self.camera.block_detections)
                n = len(self.camera.block_detections.get(color))
                for i in range(n):
                    block = self.camera.block_detections[color][i]
                    T = np.eye(4)

                    T[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T[0, 3] = block[0][0]/1000
                    T[1, 3] = block[0][1]/1000
                    T[2, 3] = block[0][2]/1000 + 0.12


                    #Go to block
                    point_over_block = self.rxarm.get_inverse(T)
                    self.rxarm.set_positions(point_over_block)
                    time.sleep(1.5)

                    #Grab it
                    T1 = np.eye(4)
                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T1[0, 3] = block[0][0]/1000
                    T1[1, 3] = block[0][1]/1000
                    T1[2, 3] = block[0][2]/1000 + 0.007

                    point_block = self.rxarm.get_inverse(T1)
                    rotated = point_block
                    rotated[4] = rotated[4] - (np.pi/2 - block[1]*np.pi/180)
                    self.rxarm.set_positions(rotated)
                    time.sleep(2)

                    self.rxarm.gripper.grasp()
                    time.sleep(1)
                    self.rxarm.gripper.release()
                    time.sleep(1)

                    rotated2 = rotated
                    rotated2[4] = rotated2[4] - (np.pi/2)
                    self.rxarm.set_positions(rotated2)
                    time.sleep(2)

                    self.rxarm.gripper.grasp()
                    time.sleep(3)
                    
                    #Go back up
                    self.rxarm.set_positions(point_over_block)
                    time.sleep(2)

                    #Go to Drop Position
                    drop_colour = drop_points_colors[bl_nb]
                    bl_nb+=1

                    T[0, 3] = drop_colour[0]/1000
                    T[1, 3] = drop_colour[1]/1000
                    T[2, 3] = drop_colour[2]/1000 + 0.04

                    T1[:3, :3]  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    T1[0, 3] = drop_colour[0]/1000
                    T1[1, 3] = drop_colour[1]/1000
                    T1[2, 3] = drop_colour[2]/1000 + 0.1

                    drop_over_position = self.rxarm.get_inverse(T1)

                    rotated_drop_over = drop_over_position
                    rotated_drop_over[4] = rotated_drop_over[4] - (np.pi/2)
                    self.rxarm.set_positions(rotated_drop_over)
                    time.sleep(2)

                    # self.rxarm.set_positions(drop_over_position)
                    # time.sleep(2)

                    drop_position = self.rxarm.get_inverse(T)
                    rotated_drop = drop_position
                    rotated_drop[4] = rotated_drop[4] - (np.pi/2)
                    self.rxarm.set_positions(rotated_drop)
                    time.sleep(2)


                    # self.rxarm.set_positions(drop_position)
                    time.sleep(2)
                    self.rxarm.gripper.release()
                    time.sleep(2)

                    #Go back up

                    self.rxarm.set_positions(rotated_drop_over)
                    time.sleep(2)

    def motion_planning(self):
        obstacle_list = [motion_planner.Obstacle(np.array([0.075, -0.05, 0]), _r=0.03, _h=0.16),
                         motion_planner.Obstacle(np.array([ -0.075,0.35, 0]), _r=0.03, _h=0.16),]
        
        start = np.array([0,0,0,0,0])
        # goal = np.array([90,-60,100,90,0])
        # goal = np.array([-1.50748957, -0.31914627,  0.70101839,  1.18892421, -1.50748957])/D2R
        goal = np.array([1.11554625, 0.2174128 , 0.04183562, 1.3115479 , 1.11554625])/D2R
        # goal = np.array([89, 0, 0, 0, 0])
        test = False
        rrt = motion_planner.RRT(start, goal, obstacle_list, test)
        path = rrt.planning()
        print(path)
        for i in range(len(path)-1,-1,-5):
            self.rxarm.set_positions(path[i].tolist())
            time.sleep(0.5)
        self.next_state = "idle"


    def record_open(self):
        self.status_message = "State: Record Waypoints - Recording waypoints"
        self.current_state = "record"
        self.next_state = "idle"
        # pos = [-np.pi/2, -0.5, -0.3, 0.0, 0.0]
        # self.rxarm.set_positions(pos)
        self.recorded_waypoints.append([self.rxarm.get_positions().tolist(), 1])
        print(f'This waypoint: {self.rxarm.get_positions().tolist()} \nUpdated list of waypoints: {self.recorded_waypoints}')

    def record_closed(self):
        self.status_message = "State: Record Waypoints - Recording waypoints"
        self.current_state = "record"
        self.next_state = "idle"
        # pos = [-np.pi/2, -0.5, -0.3, 0.0, 0.0]
        # self.rxarm.set_positions(pos
        self.recorded_waypoints.append([self.rxarm.get_positions().tolist(), 0])
        print(f'This waypoint: {self.rxarm.get_positions().tolist()} \nUpdated list of waypoints: {self.recorded_waypoints}')

    def clear(self):
        root = tkinter.Tk()
        root.title("Double checker for the clear button so you don't" 
                   "accidently clear your precious waypoints")
        root.geometry('50x50')
        decision = tkinter.messagebox.askyesno(message='Do you want to clear the waypoints?')
        if decision == YES:
            self.recorded_waypoints = []
            root.destroy()
        else:
            root.destroy()
        root.mainloop()

        print(f'Updated list of waypoints: {self.recorded_waypoints}')
        self.next_state = "idle"
    
    # Helper function
    def find_delta_points(self, prev_pt, curr_pt):
        delta_pt = []
        max_delta_angle = -1
        max_angle_ind = 0
        for i in range(len(prev_pt)):
            delta_angle = abs(curr_pt[i] - prev_pt[i])
            delta_pt.append(delta_angle)
            if max_delta_angle < delta_angle:
                max_delta_angle = delta_angle
                max_angle_ind = i
        return delta_pt, max_delta_angle, max_angle_ind

    def execute_record(self):
        """!
        @brief      Go through all the recorded waypoints
        """
        self.status_message = "State: Execute Recorded - Executing motion plan of the recorded waypoints"
        self.current_state = "execute_record"
        self.next_state = "idle"
        prev_point = self.rxarm.get_positions().tolist()
        delta_point = []
        max_speed = np.pi/2 #rad/seconds
        print(self.recorded_waypoints)
        

        for point in self.recorded_waypoints:
            delta_pt, max_delta_angle, max_angle_ind = self.find_delta_points(prev_point,point[0])
            time_max_speed = max_delta_angle - 0.2*np.pi
            if time_max_speed < 0:
                time_max_speed = 0
            moving_time = time_max_speed/max_speed + 0.8 
            
            accel_time = 0.4
            self.rxarm.set_moving_time(moving_time)
            self.rxarm.set_accel_time(accel_time)

            self.rxarm.set_positions(point[0])
            time.sleep(moving_time + 0.1)
            if point[1] == 0:
                self.rxarm.gripper.grasp()
                time.sleep(0.5)
            elif point[1] == 1:
                self.rxarm.gripper.release()
                time.sleep(0.5)
            prev_point = point[0]




    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"


class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)