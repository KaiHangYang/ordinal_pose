import numpy as np

h36m_root = 0
h36m_pose = np.array([
    [9, 10], [8, 9],
    [8, 14], [14, 15], [15, 16],
    [8, 11], [11, 12], [12, 13],
    [7, 8], [0, 7],
    [0, 4], [4, 5], [5, 6],
    [0, 1], [1, 2], [2, 3]
])
h36m_bone_colors = np.array([
    [0, 255, 0], [0, 255, 0],
    [0, 0, 255], [0, 0, 255], [0, 0, 255],
    [255, 153, 0], [255, 153, 0], [255, 153, 0],
    [0, 128, 0], [0, 255, 0],
    [0, 0, 255], [0, 0, 255], [0, 0, 255],
    [255, 153, 0], [255, 153, 0], [255, 153, 0],
])

h36m_joint_colors = np.array([
    [214, 10, 153],
    [13, 17, 0],
    [13, 10, 100],
    [153, 13, 10],
    [13, 151, 100],
    [10, 13, 150],
    [151, 13, 0],
    [19, 12, 100],
    [121, 13, 52],
    [6, 13, 12],
    [134, 13, 13],
    [29, 112, 13],
    [221, 51, 13],
    [29, 12, 120],
    [11, 13, 152],
    [191, 33, 52],
    [116, 13, 212]
])
