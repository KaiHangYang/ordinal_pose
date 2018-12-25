import numpy as np
import math

def axisangle2matrix(axis, angle):
    axis = normalize(axis)
    half_angle = angle / 2

    a = math.cos(half_angle)
    b = math.sin(half_angle) * axis[0]
    c = math.sin(half_angle) * axis[1]
    d = math.sin(half_angle) * axis[2]

    mat = np.array([
        [1 - 2*c**2 - 2*d**2, 2*b*c - 2*a*d, 2*a*c + 2*b*d],
        [2*b*c + 2*a*d, 1 - 2*b**2 - 2*d**2, 2*c*d - 2*a*b],
        [2*b*d - 2*a*c, 2*a*b + 2*c*d, 1 - 2*b**2 - 2*c**2],
        ])

    return mat
# https://zhuanlan.zhihu.com/p/45404840
def matrix2euler(mat):
    # mat is 3x3
    # default is the yaw(y)*pitch(x)*roll(z) convention
    ### handle the Gimbal Lock ###
    # mat[1][2] is -sin(pitch_angle) 
    if mat[1][2] == 1:
        # pitch_angle == -pi/2
        pitch_angle = -math.pi / 2
        yaw_angle = 0
        roll_angle = math.atan2(-mat[0][1], mat[0][0]) - yaw_angle
    elif mat[1][2] == -1:
        # pitch_angle == pi / 2
        pitch_angle = math.pi / 2
        roll_angle = 0
        yaw_angle = math.atan2(mat[0][1], mat[0][0]) + roll_angle
    else:
        yaw_angle = math.atan2(mat[0][2], mat[2][2]) # y axis
        pitch_angle = math.asin(-mat[1][2])
        roll_angle = math.atan2(mat[1][0], mat[1][1])

    return pitch_angle, yaw_angle, roll_angle


def euler2matrix(pitch_angle, yaw_angle, roll_angle):
    # default is the yaw(y) pitch(x) roll(z) convention
    x_angle, y_angle, z_angle = pitch_angle, yaw_angle, roll_angle
    cos_x = math.cos(x_angle)
    sin_x = math.sin(x_angle)

    cos_y = math.cos(y_angle)
    sin_y = math.sin(y_angle)

    cos_z = math.cos(z_angle)
    sin_z = math.sin(z_angle)

    mat_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
        ])

    mat_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
        ])

    mat_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
        ])

    return np.dot(mat_y, np.dot(mat_x, mat_z))

def normalize(vec):
    vec = np.array(vec)
    vec = vec / (np.linalg.norm(vec) + 0.000001)
    return vec
