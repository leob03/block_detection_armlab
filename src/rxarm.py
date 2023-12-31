"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from functools import partial
from kinematics import FK_dh, FK_pox, get_pose_from_T, IK_geometric
import time
import csv
import sys, os

from builtins import super
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from resource.config_parse import parse_dh_param_file
from sensor_msgs.msg import JointState
import rclpy

sys.path.append('../../interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot') 
from arm import InterbotixManipulatorXS
from mr_descriptions import ModernRoboticsDescription as mrd

"""
TODO: Implement the missing functions and add anything you need to support them
"""
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixManipulatorXS):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """
    def __init__(self, dh_config_file=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dh_config_file  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_model="rx200")
        self.joint_names = self.arm.group_info.joint_names
        self.num_joints = 5
        # Gripper
        self.gripper_state = True
        # State
        self.initialized = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 2.0
        self.accel_time = 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None
        # DH Params
        self.dh_params = []
        self.dh_config_file = dh_config_file
        if (dh_config_file is not None):
            self.dh_params = RXArm.parse_dh_param_file(dh_config_file)
        #POX params
        self.M_matrix = []
        self.S_list = []

    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        time.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1

        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 2.0
        self.accel_time = 0.5
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.gripper.release()
        self.initialized = True
        return self.initialized

    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.arm.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False

    def set_positions(self, joint_positions):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        self.arm.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=False)

    def set_moving_time(self, moving_time):
        self.moving_time = moving_time

    def set_accel_time(self, accel_time):
        self.accel_time = accel_time

    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 0)

    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 1)

    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        return self.position_fb

    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb

    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb
    
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6
    
    def rotationMatrixToEulerAngles(self, R) :
        if not (self.isRotationMatrix(R)):
            return np.array([0,0,0])
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = np.arctan2(R[2,1] , R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else :
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
    
        return np.array([x, y, z])

    # @_ensure_initialized
    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose.

        @return     The EE pose as [x, y, z, phi] or as needed.
        """
        #Hannah
        # joint_angles = self.position_fb
    
        # joint_link = 2
        # print(joint_angles, joint_link)
        # H = FK_dh(joint_angles, joint_link)
        # twist_coordinates = np.array([[0.0,      0.0,    0.0,   0.0, 0.0, 1.0],
        #                               [-104.57,  0.0,    0.0,   0.0, 1.0, 0.0],
        #                               [-304.57,  0.0,    50.0,  0.0, 1.0, 0.0],
        #                               [-304.57,  0.0,    250.0, 0.0, 1.0, 0.0],
        #                               [0.0,      304.57, 0.0,   1.0, 0.0, 0.0]])
        
        joint_angles = self.position_fb
        # print(joint_angles)
        twist_coordinates = np.array([[0.0,     0.0,      0.0,    0.0, 0.0, 1.0],
                                      [0.0,    -0.10457,  0.0,   -1.0, 0.0, 0.0],
                                      [0.0,    -0.30457,  0.05,  -1.0, 0.0, 0.0],
                                      [0.0,    -0.30457,  0.25,  -1.0, 0.0, 0.0],
                                      [-0.30457, 0.0,      0.0,    0.0, 1.0, 0.0]])
        # twist_coordinates = np.array([[0.0,     0.0,      0.0,    0.0, 0.0, 1.0],
        #                               [0.0,    -104.57,  0.0,   -1.0, 0.0, 0.0],
        #                               [0.0,    -304.57,  50,  -1.0, 0.0, 0.0],
        #                               [0.0,    -304.57,  250,  -1.0, 0.0, 0.0],
        #                               [-304.57, 0.0,      0.0,    0.0, 1.0, 0.0]])
        m_mat = np.array([[0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.42415],
                          [1.0, 0.0, 0.0, 0.30457],
                          [0.0, 0.0, 0.0, 1.0]])
        H = FK_pox(joint_angles, m_mat, twist_coordinates)
        # phi, theta, psi = self.rotationMatrixToEulerAngles(H[:3,:3])
        r = Rotation.from_matrix(H[:3,:3])
        # phi, theta, psi = r.as_euler("zyz", degrees=True)
        phi, theta, psi = H[0:3, 2]
        # phi, theta, psi =  [0.1, 0.1, 0.1]
        return [H[0][3], H[1][3], H[2][3], phi, theta, psi]
        # return [H[0][3], H[1][3], H[2][3], r[0], r[1], r[2]]

    def get_naive_waypoints(self, T):
        waypoints_grab = []
        # hover_123 = IK_geometric(T)
        # hover = np.array([hover_123[0], hover_123[1], hover_123[2], np.pi/2, 0])
        hover = IK_geometric(T)
        waypoints_grab.append(hover.tolist())
        T[2,3] = T[2,3] - 0.1
        # destination_123 = IK_geometric(T)
        # destination = np.array([destination_123[0], destination_123[1], destination_123[2], np.pi/2, 0])
        destination = IK_geometric(T)
        waypoints_grab.append(destination.tolist())

        # if np.size(hover) == 0 or np.size(destination) == 0:
        #     return waypoints_grab, False
        # else:
        #     return waypoints_grab, True
        return waypoints_grab
    
    def get_inverse(self, T):
        # return IK_geometric(x,y,z,phi,theta).tolist()
        full_traj = IK_geometric(T).tolist()
        # print(len(full_traj))
        # selected_traj = [full_traj[i] for i in range(0, len(full_traj), 4)]
        # print(len(selected_traj))
        return full_traj
    

    @_ensure_initialized
    def get_wrist_pose(self):
        """!
        @brief      TODO Get the wrist pose.

        @return     The wrist pose as [x, y, z, phi] or as needed.
        """
        return [0, 0, 0, 0]

    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise 
        """
        return -1

    def parse_dh_param_file(self):
        print("Parsing DH config file...")
        dh_params = parse_dh_param_file(self.dh_config_file)
        print("DH config file parse exit.")
        return dh_params

    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params
    



class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        self.node = rclpy.create_node('rxarm_thread')
        self.subscription = self.node.create_subscription(
            JointState,
            '/rx200/joint_states',
            self.callback,
            10
        )
        self.subscription  # prevent unused variable warning
        rclpy.spin_once(self.node, timeout_sec=0.5)

    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5]
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose())
        #for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_motor_pid_params(name)))
        if (__name__ == '__main__'):
            print(self.rxarm.position_fb)

    def run(self):
        """!
        @brief      Updates the RXArm Joints at a set rate if the RXArm is initialized.
        """
        while True:
            rclpy.spin_once(self.node) 
            time.sleep(0.02)


if __name__ == '__main__':
    rclpy.init() # for test
    rxarm = RXArm()
    print(rxarm.joint_names)
    armThread = RXArmThread(rxarm)
    armThread.start()
    try:
        joint_positions = [-1.0, 0.5, 0.5, 0, 1.57]
        rxarm.initialize()

        rxarm.arm.go_to_home_pose()
        rxarm.set_gripper_pressure(0.5)
        rxarm.set_joint_positions(joint_positions,
                                  moving_time=2.0,
                                  accel_time=0.5,
                                  blocking=True)
        rxarm.gripper.grasp()
        rxarm.arm.go_to_home_pose()
        rxarm.gripper.release()
        rxarm.sleep()

    except KeyboardInterrupt:
        print("Shutting down")

    rclpy.shutdown()