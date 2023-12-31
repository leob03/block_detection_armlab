"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R
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
    # H = np.eye(4)
    # links = np.shape(s_lst)[0]
    # # joint_angles = np.array([0.0,  np.pi/2,    0.0,   0.0, 0.0, 0.0])
    # for i in range(links):
    #     # print(s_lst[i,:])
    #     twisted = joint_angles[i]*s_lst[i,:]
    #     # print(twisted)
    #     # print(expm(hat(twisted)))
    #     # H = np.dot(H, expm(hat(twisted)))
    #     H = H @ expm(hat(twisted))
    #     # H = np.dot(H, get_expm(s_lst[i,:], joint_angles[i]))

    # # return np.dot(m_mat, H)
    # # print(H)
    # return H@m_mat

    H = np.eye(4)
    links = np.shape(s_lst)[0]
    # joint_angles = np.array([0.0,  np.pi/2,    0.0,   0.0, 0.0, 0.0])
    for i in range(links):
        # print(s_lst[i,:])
        twisted = joint_angles[i]*s_lst[i,:]
        H = H @ expm(hat(twisted))
    H = H @ m_mat
    # H = H @ rot
    # H = rot @ H
    # return np.dot(m_mat, H)
    # print(H)
    return H



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


# def IK_geometric(dh_params, pose):
# def IK_geometric(x, y, z, phi, theta):
def IK_geometric(T):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """

    l1 = 0.10457               
    l2 = np.sqrt(0.2**2+0.05**2) 
    l3 = 0.2                   
    l4 = 158.575/1000   
    offset = np.arctan2(0.05, 0.2) 

    p_e = T[0:3, 3]
    # p_e = np.array([x, y, z])
    # p_e.reshape(3,1)
    R_e = T[0:3, 0:3]
    a_e = R_e[0:3, 2]
    # R_e = (R.from_euler('x', -90-phi, degrees=True)*R.from_euler('z', theta, degrees=True)).as_matrix()
    # a_e = R_e[0:3, 2]
    # a_e.reshape(3,1)
    # print(np.shape(a_e))
    # x = l4*a_e
    # x.reshape(3,1)
    # p_w_ = p_e - (l4*a_e).reshape(3,1)
    # p_w = np.array([p_w_[0][0],p_w_[1][0],p_w_[2][0]])
    # print(x.shape, p_w.shape)
    # p_w = np.array([x-l4*a_e[0], y-l4*a_e[1], z-l4*a_e[2]])
    # print('p:', p_w)
    # theta1 = np.arctan2(p_w[1], p_w[0]) ## + np.pi/2
    p_w = p_e - (l4*a_e)
    theta1 = np.arctan2(-p_w[0], p_w[1])
    while (theta1 > np.pi):
        theta1 -= 2*np.pi
    while (theta1 < -np.pi):
        theta1 += 2*np.pi

    p_wx = p_w[0]
    p_wy = p_w[1]
    p_wz = p_w[2]
    print(f'\nx: {p_wx}\ny: {p_wy}\nz: {p_wz}')
    r_w = np.sqrt(p_wx**2 + p_wy**2)

    # if p_wz < 20/1000:
    #     p_wz = 20/1000

    theta3 = -np.arccos((p_wx**2 + p_wy**2 + (p_wz-l1)**2 - l2**2 - l3**2)/(2*l2*l3)) ## or negative
    theta2 = np.arctan2(p_wz-l1, np.sqrt(p_wx**2 + p_wy**2)) - np.arctan2(l3*np.sin(theta3), l2 + l3*np.cos(theta3))
    # theta2 = np.arctan2((l2 + l3*np.cos(theta3))*p_wz - l3*np.sin(theta3)*r_w, (l2 + l3*np.cos(theta3))*r_w + l3*np.sin(theta3)*p_wz)
    theta3 = -np.pi/2 + offset - theta3
    # theta2 = t_offset - theta2
    theta2 = np.pi/2 - offset - theta2
    theta5 = theta1
    theta4 = np.pi/2 - theta2 - theta3

    max_wrist_len = 405.73/1000
    # if np.sqrt((p_wz - 103.91/1000)**2 + p_wx**2 + p_wy**2) > max_wrist_len:
    #     print("\n----------------Location outside of boundaries.------------------\n")
    #     return np.array([])


    # # Wrist Perpendicular
    # if p_wz <= 100/1000:
    #     theta4 = np.pi/2 - theta2 - theta3
    #     if np.sqrt((p_wz + 154.15/1000-103.91/1000)**2 + p_wx**2 + p_wy**2) > max_wrist_len:
    #         print("\n----------------Location outside of boundaries.------------------\n")
    #         return np.array([])

    # # Wrist Parallel
    # else:
    #     theta4= -theta2 - theta3
    #     if np.sqrt((np.sqrt(p_wx**2 + p_wy**2) - 154.15/1000)**2 + (p_wz - 103.91/1000)**2) > max_wrist_len:
    #         print("\n----------------Location outside of boundaries.------------------\n")
    #         return np.array([])
        # if (np.sqrt(p_wx**2 + p_wy**2) - 154.15/1000)**2 < 100/1000:
        #     print("\n----------------Arm is too close to base.------------------\n")
        #     return np.array([])

    # theta3 = -np.arccos((p_wx**2 + p_wy**2 + (p_wz-l1)**2 - l2**2 - l3**2)/(2*l2*l3)) ## or negative
    # theta2 = np.arctan2(p_wz-l1, np.sqrt(p_wx**2 + p_wy**2)) - np.arctan2(l3*np.sin(theta3), l2 + l3*np.cos(theta3))
    # # theta2 = np.arctan2((l2 + l3*np.cos(theta3))*p_wz - l3*np.sin(theta3)*r_w, (l2 + l3*np.cos(theta3))*r_w + l3*np.sin(theta3)*p_wz)
    # theta3 = -np.pi/2 + offset - theta3
    # # theta2 = t_offset - theta2
    # theta2 = np.pi/2 - offset - theta2
    # theta4 = np.pi/2 - theta2 - theta3
    # theta5 = -theta1

    return np.array([theta1, theta2, theta3, theta4, theta5])

