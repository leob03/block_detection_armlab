"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
from tkinter import *
import tkinter.messagebox

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

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        self.status_message = "Calibration - Completed Calibration"


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

    # def yesClear(self, root):
    #     tkinter.messagebox.showinfo("Ok, clearing!")
    #     self.recorded_waypoints = []
    #     root.destroy()

    # def noClear(self, root):
    #     tkinter.messagebox.showinfo("Yep, sounds good, not clearing!")
    #     root.destroy()

    def clear(self):
        # root = tkinter.Tk()
        # root.title("Double checker for the clear button so you don't" 
        #            "accidently clear your precious waypoints")
        # root.geometry('1000x300')
        # ButtonYes = Button(root, text = "YES, I WANT TO CLEAR", command = self.yesClear(root), height = 5, width = 25)
        # ButtonNo = Button(root, text = "NO, I DON'T WANT TO CLEAR!!!", command = self.noClear(root), height = 5, width = 25)
        # ButtonYes.pack(side = 'right')
        # ButtonNo.pack(side = 'left')
        # root.mainloop()

        self.recorded_waypoints = []
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