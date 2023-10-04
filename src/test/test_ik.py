import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm

def ik(T):
#Law of Cosines for the displaced link
    p_e = T[0:3,3]
    R_e = T[0:3, 0:3]
    a_e = R_e[0:3, 2]
    # p_w = p_e - 
    l1 = 200
    l2 = 50
    L1 = 205.73
    L2 = 200
    z0 = 103.91
    gamma1 = np.arccos((-l2^2 + l1^2 + L1^2)/(2*l1*L1))
    gamma2 = np.arccos((-l1^2 + l2^2 + L1^2)/(2*l2*L1))

    ## Variables common between positions based on inputs
    # h, K, alpha, and beta are the same across all 4 possibilities
    h = np.sqrt(x^2 + y^2 + (z - z0)^2)
    k = np.arctan(z/(np.sqrt(x^2 + y^2)))
    beta = np.arccos((-h^2 + L1^2 + L2^2)/(2*L1*L2))
    alpha = np.arccos((-L2^2 + L1^2 + h^2)/(2*L1*h))

    ## 1 - position of o_c
    theta1_1 = np.pi + -np.arctan(y/x)
    theta2_1 = -(np.pi/2 - k + alpha + gamma1)
    theta3_1 = np.pi - beta - gamma2

    ## 2 - position of o_c
    theta1_2 = -np.arctan(y/x)
    theta2_2 = np.pi/2 - k + alpha - gamma1
    theta3_2 = np.pi - (beta - gamma2)

    ## 3 - position of o_c
    theta1_3 = np.pi + -np.arctan(y/x)
    theta2_3 = -(np.pi/2 - k - alpha + gamma1)
    theta3_3 = -(2*np.pi - beta + gamma2)

    ## 4 - position of o_c  
    theta1_4 = -np.arctan(y/x)
    theta2_4 = np.pi/2 - gamma1 - alpha - k
    theta3_4 = -(gamma2 + beta - np.pi)
    printer = gamma2 + beta

    possible_pos = np.array([theta1_1,theta2_1,theta3_1],[theta1_2,theta2_2,theta3_2],[theta1_3,theta2_3,theta3_3],[theta1_4,theta2_4,theta3_4])