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
def IK_geometric(T):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    # ## inputs
    # # x, y, z inputs
    # # x =
    # # y =
    # # z = 

    # # # theta(5), phi(6) inputs
    # # theta = 
    # # phi =

    # #Law of Cosines for the displaced link
    # l1 = 200
    # l2 = 50
    # L1 = 205.73
    # L2 = 200
    # z0 = 103.91
    # gamma1 = np.arccos((-l2^2 + l1^2 + L1^2)/(2*l1*L1))
    # gamma2 = np.arccos((-l1^2 + l2^2 + L1^2)/(2*l2*L1))

    # ## Variables common between positions based on inputs
    # # h, K, alpha, and beta are the same across all 4 possibilities
    # h = np.sqrt(x^2 + y^2 + (z - z0)^2)
    # k = np.arctan(z/(np.sqrt(x^2 + y^2)))
    # beta = np.arccos((-h^2 + L1^2 + L2^2)/(2*L1*L2))
    # alpha = np.arccos((-L2^2 + L1^2 + h^2)/(2*L1*h))

    # ## 1 - position of o_c
    # theta1_1 = np.pi + -np.arctan(x/y)
    # theta2_1 = -(np.pi/2 - k + alpha + gamma1)
    # theta3_1 = np.pi - beta - gamma2

    # ## 2 - position of o_c
    # theta1_2 = -np.arctan(x/y)
    # theta2_2 = np.pi/2 - k + alpha - gamma1
    # theta3_2 = np.pi - (beta - gamma2)

    # ## 3 - position of o_c
    # theta1_3 = np.pi + -np.arctan(x/y)
    # theta2_3 = -(np.pi/2 - k - alpha + gamma1)
    # theta3_3 = -(2*np.pi - beta + gamma2)

    # ## 4 - position of o_c  
    # theta1_4 = -np.arctan(x/y)
    # theta2_4 = np.pi/2 - gamma1 - alpha - k
    # theta3_4 = -(gamma2 + beta - np.pi)
    # printer = gamma2 + beta

    # possible_pos = np.array([theta1_1,theta2_1,theta3_1],[theta1_2,theta2_2,theta3_2],[theta1_3,theta2_3,theta3_3],[theta1_4,theta2_4,theta3_4])

    l1 = 0.10457               
    l2 = np.sqrt(0.2**2+0.05**2) 
    l3 = 0.2                   
    l4 = 158.575/1000   
    offset = np.arctan2(0.05, 0.2) 

    p_e = T[0:3, 3]
    R_e = T[0:3, 0:3]
    a_e = R_e[0:3, 2]

    p_w = p_e - l4*a_e
    # p_w = np.array([-0.125, 0.35, 0.152])
    print(p_w)
    # theta1 = np.arctan2(p_w[1], p_w[0]) ## + np.pi/2
    theta1 = np.arctan2(-p_w[0], p_w[1])
    while (theta1 > np.pi):
        theta1 -= 2*np.pi
    while (theta1 < -np.pi):
        theta1 += 2*np.pi



    p_wx = p_w[0]
    p_wy = p_w[1]
    p_wz = p_w[2]
    r_w = np.sqrt(p_wx**2 + p_wy**2)
    theta3 = -np.arccos((p_wx**2 + p_wy**2 + (p_wz-l1)**2 - l2**2 - l3**2)/(2*l2*l3)) ## or negative
    theta2 = np.arctan2(p_wz-l1, np.sqrt(p_wx**2 + p_wy**2)) - np.arctan2(l3*np.sin(theta3), l2 + l3*np.cos(theta3))
    # theta2 = np.arctan2((l2 + l3*np.cos(theta3))*p_wz - l3*np.sin(theta3)*r_w, (l2 + l3*np.cos(theta3))*r_w + l3*np.sin(theta3)*p_wz)
    theta3 = -np.pi/2 + offset - theta3
    # theta2 = t_offset - theta2
    theta2 = np.pi/2 - offset - theta2
    theta4 = np.pi/2 - theta2 - theta3
    theta5 = -theta1

    return np.array([theta1, theta2, theta3, theta4, theta5])

