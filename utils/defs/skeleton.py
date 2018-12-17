import numpy as np

class mSkeleton15(object):
    skeleton_index = 1
    n_joints = 15
    n_bones = 14
    head_indices = np.array([7, 8])
    h36m_selected_index = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16])
    bone_indices = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]])
    flip_array = np.array([[1, 4], [2, 5], [3, 6], [9, 12], [10, 13], [11, 14]])
    bone_colors = np.array([[1.000000, 1.000000, 0.000000], [0.492543, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [1.000000, 0.349454, 0.000000], [0.499439, 0.558884, 1.000000], [0.000000, 0.362774, 0.000000], [0.500312, 0.000000, 0.624406], [0.000000, 1.000000, 1.000000], [1.000000, 0.499433, 0.611793], [1.000000, 0.800000, 1.000000], [0.000000, 0.502502, 0.611632], [0.200000, 0.700000, 0.300000], [0.700000, 0.300000, 0.100000], [0.300000, 0.200000, 0.800000]])
    joint_names = np.array([
        "root",
        "right hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "spin",
        "head",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_shoulder",
        "left_elbow",
        "left_wrist"
    ])
    # definations for dlcm
    level_structure = [
        # level 0
        np.arange(n_joints).astype(np.int32),
        # level 1
        bone_indices,
        # level 2,
        np.array([[1, 2], [4, 5], [0, 3], [6, 7], [9, 10], [12, 13], [8, 11]])
    ]
    level_n = len(level_structure)
    level_nparts = [len(i) for i in level_structure]

class mSkeleton17(object):
    skeleton_index = 0
    n_joints = 17
    n_bones = 16
    head_indices = np.array([8, 10])
    h36m_selected_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    bone_indices = np.array([[0, 1],[1, 2],[2, 3],[0, 4],[4, 5],[5, 6],[0, 7],[7, 8],[8, 9],[9, 10],[8, 11],[11, 12],[12, 13],[8, 14],[14, 15],[15, 16]])
    flip_array = np.array([[11, 14], [12, 15], [13, 16], [1, 4], [2, 5], [3, 6]])
    bone_colors = np.array([[1.000000, 1.000000, 0.000000],[0.492543, 0.000000, 0.000000],[0.000000, 1.000000, 0.000000],[1.000000, 0.349454, 0.000000],[0.499439, 0.558884, 1.000000],[0.000000, 0.362774, 0.000000],[0.500312, 0.000000, 0.624406],[0.501744, 0.724322, 0.275356],[0.000000, 1.000000, 1.000000],[1.000000, 0.000000, 1.000000],[1.000000, 0.499433, 0.611793],[1.000000, 0.800000, 1.000000],[0.000000, 0.502502, 0.611632],[0.200000, 0.700000, 0.300000],[0.700000, 0.300000, 0.100000],[0.300000, 0.200000, 0.800000]])
    joint_names = np.array([
        "root",
        "right hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "mid_spin",
        "throat",
        "nose",
        "back_head",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_shoulder",
        "left_elbow",
        "left_wrist"
    ])


class mSkeleton16(object):
    skeleton_index = 2
    n_joints = 16
    n_bones = 15

    score_joints_idx = np.array([0, 1, 2, 3, 4, 5, 10, 11, 14, 15]) # the joints to compute accuracy according to the code of DLCM
    bone_indices = np.array([[0, 1], [1, 2], [6, 2], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15]])
    flip_array = np.array([[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]])
    bone_colors = np.array([[1.000000, 1.000000, 0.000000], [0.492543, 0.000000, 0.000000], [0.492543, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [1.000000, 0.349454, 0.000000], [0.499439, 0.558884, 1.000000], [0.000000, 0.362774, 0.000000], [0.500312, 0.000000, 0.624406], [0.000000, 1.000000, 1.000000], [1.000000, 0.499433, 0.611793], [1.000000, 0.800000, 1.000000], [0.000000, 0.502502, 0.611632], [0.200000, 0.700000, 0.300000], [0.700000, 0.300000, 0.100000], [0.300000, 0.200000, 0.800000]])
    joint_names = np.array([
        "r ankle",
        "r knee",
        "r hip",
        "l hip",
        "l knee",
        "l ankle",
        "pelvis",
        "thorax",
        "upper neck",
        "head top",
        "r wrist",
        "r elbow",
        "r shoulder",
        "l shoulder",
        "l elbow",
        "l wrist"
    ])
    # definations for dlcm
    level_structure = [
        # level 0
        np.arange(n_joints).astype(np.int32),
        # level 1
        np.array([[0, 1], [1, 2], [3, 4], [4, 5], [2, 6], [3, 6], [7, 8], [8, 9], [10, 11], [11, 12], [13, 14], [14, 15]]),
        # level 2,
        np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
    ]
    level_n = len(level_structure)
    level_nparts = [len(i) for i in level_structure]
