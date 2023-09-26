"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format 
                              The one they want is [a, alpha, d, theta]
                              The one I have is [theta, d, a, alpha]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from (denoted by a number 0-4)

    @return     a transformation matrix representing the pose of the desired link
    """
    angle1 = joint_angles[0] % 2*np.pi
    angle2 = -joint_angles[1] % 2*np.pi
    angle3 = -joint_angles[2] % 2*np.pi
    # angle3 = joint_angles[2] - np.pi/2
    angle4 = -joint_angles[3] % 2*np.pi
    angle5 = joint_angles[4] % 2*np.pi

    print('Angle1:', angle1)
    print('Angle2:', angle2)
    print('Angle3:', angle3)
    # print('Angle4:', angle4)
    # print('Angle5:', angle5)


    # dhp = np.array([[angle1, 103.91, 0, np.pi/2],
    #                       [angle2 + np.pi/2, 0, 200, 0],
    #                       [-np.pi/2, 0, 50, 0],
    #                       [angle3, 0, 200, 0],
    #                       [angle4 + np.pi/2, 0, 0, np.pi/2],
    #                       [angle5 - np.pi/2, 66 + 65, 0, 0]])


    dhp = np.array([[angle1, 103.91, 0, np.pi/2],
                    [angle2 + np.pi/2, 0, 200, 0],
                    [-np.pi/2, 0, 50, 0],
                    [angle3, 0, 200, 0],
                    [angle4 + np.pi/2, 0, 0, -np.pi/2],
                    [angle5 - np.pi/2, 66 + 65, 0, 0]])

    H = np.eye(4) # List of all of the As
    count = 1
    count2 = False
    for param in dhp:
        A_i = np.array([[np.cos(param[0]), -np.sin(param[0])*np.cos(param[3]), np.sin(param[0])*np.sin(param[3]), param[2]*np.cos(param[0])],
                        [np.sin(param[0]), np.cos(param[0])*np.cos(param[3]), -np.cos(param[0])*np.sin(param[3]), param[2]*np.sin(param[0])],
                        [0, np.sin(param[2]), np.cos(param[2]), param[1]],
                        [0, 0, 0, 1]])

        H = np.dot(H,A_i)
        if count == 2 and count2 == False:
            count2 = True
            break
        count += 1
        if count > link:
            break
    return H


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    pass


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this fnction return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    pass


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    pass


def hat(twist):
        Vx, Vy, Vz, Wx, Wy, Wz = twist
        return np.array([[0, -Wz, Wy, Vx],
                        [Wz, 0, -Wx, Vy],
                        [-Wy, Wx, 0, Vz],
                        [0, 0, 0, 0]])

def get_expm(twist, theta):
    twist_hat = hat(twist)
    w_hat = twist_hat[0:3,0:3]
    v = twist_hat[0:3,3]
    e_wtheta = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * np.dot(w_hat, w_hat)
    e_xitheta = np.r_[np.c_[e_wtheta, np.dot(np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * np.dot(w_hat,w_hat), v)],
                           [[0, 0, 0, 1]]]

    return e_xitheta


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    H = np.eye(4)
    links = np.shape(s_lst)[0]
    # joint_angles = np.array([0.0,  np.pi/2,    0.0,   0.0, 0.0, 0.0])
    for i in range(links):
        # print(s_lst[i,:])
        twisted = joint_angles[i]*s_lst[i,:]
        # print(twisted)
        # print(expm(hat(twisted)))
        # H = np.dot(H, expm(hat(twisted)))
        H = H @ expm(hat(twisted))
        # H = np.dot(H, get_expm(s_lst[i,:], joint_angles[i]))

    # return np.dot(m_mat, H)
    # print(H)
    return H@m_mat


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass