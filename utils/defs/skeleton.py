import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_utils import math_utils
import math

class mSkeleton15(object):
    skeleton_index = 1
    n_joints = 15
    n_bones = 14
    head_indices = np.array([7, 8])
    h36m_selected_index = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16])

    hip_indices = np.array([1, 4])

    skeleton_path = np.array([
        [0, 1, 2, 3], # right leg
        [0, 4, 5, 6], # left leg
        [0, 7, 8], # to head
        [0, 7, 9, 10, 11], # to right arm
        [0, 7, 12, 13, 14], # to left arm
    ])

    T_skeleton = np.array([
        [1, 0, 0], [0, 1, 0], [0, 1, 0],
        [-1, 0, 0], [0, 1, 0], [0, 1, 0],
        [0, -1, 0], [0, -1, 0],
        [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]
    ])

    score_joints_idx = np.arange(0, n_joints, 1) # the joints to compute accuracy according to the code of DLCM
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

    # some skeleton method
    @classmethod
    def get_bonelengths(cls, joints_3d):
        bone_lengths = []
        for cur_bone in cls.bone_indices:
            bone_lengths.append(np.linalg.norm(joints_3d[cur_bone[0]] - joints_3d[cur_bone[1]]))

        return np.array(bone_lengths)

    @classmethod
    def get_joints(cls, angles, bone_lengths):
        # for conveniouse
        bone_lengths = np.concatenate([[0], bone_lengths])
        joints_3d = np.zeros([cls.n_joints, 3])

        for cur_path in cls.skeleton_path:
            cur_mat = math_utils.euler2matrix(*angles[cur_path[0]])

            for cur_idx in range(1, len(cur_path)):
                joint_source = joints_3d[cur_path[cur_idx-1]]

                tmp_mat = math_utils.euler2matrix(*angles[cur_path[cur_idx]])
                cur_mat = np.dot(cur_mat, tmp_mat)

                raw_vec = cls.T_skeleton[cur_path[cur_idx]-1] * bone_lengths[cur_path[cur_idx]]
                joints_3d[cur_path[cur_idx]] = np.dot(cur_mat, raw_vec) + joint_source

        return joints_3d


    @classmethod
    def get_angles(cls, joints_3d):
        joints_3d = joints_3d.copy()
        assert(joints_3d.shape[0] == cls.n_joints)
        angles = np.zeros([cls.n_joints, 3])

        for cur_path in cls.skeleton_path:
            cur_mat = np.linalg.inv(math_utils.euler2matrix(*angles[cur_path[0]]))
            for cur_idx in range(1, len(cur_path)):
                joint_source = joints_3d[cur_path[cur_idx-1]]
                joint_target = joints_3d[cur_path[cur_idx]]

                # TODO waiting to think about it.
                joint_source = np.dot(cur_mat, joint_source)
                joint_target = np.dot(cur_mat, joint_target)

                cur_vec = math_utils.normalize(joint_target - joint_source)
                prev_vec = cls.T_skeleton[cur_path[cur_idx]-1]

                rotate_axis = np.cross(prev_vec, cur_vec)
                rotate_angle = np.arccos(np.clip(np.dot(cur_vec, prev_vec), -1, 1))
                if not (rotate_axis == 0).all():
                    # not parallel
                    tmp_mat = math_utils.axisangle2matrix(axis=rotate_axis, angle=rotate_angle)
                    cur_mat = np.dot(np.linalg.inv(tmp_mat), cur_mat)
                    angles[cur_path[cur_idx]] = math_utils.matrix2euler(tmp_mat)
                else:
                    angles[cur_path[cur_idx]] = np.zeros([3])

        return angles

    @classmethod
    def jitter_angles(cls, angles, jitter_size=math.pi/20):
        #TODO the index 0 is the skeleton rotation, this can be jitterred in a wide range
        assert(jitter_size >= 0)
        angles = angles.copy()
        cur_jitters = np.random.uniform(-jitter_size, jitter_size, angles.shape)

        cur_jitters[cls.hip_indices] = 0 # don't jitter the hip angles
        cur_jitters[0, 1] = np.random.uniform(-math.pi, math.pi) # jitter root around the y axis

        angles = angles + cur_jitters
        return angles

    @classmethod
    def jitter_bonelengths(cls, bone_lengths, jitter_size=20):
        # default jitter_size is 20mm
        # first, jitter the bone lengths 
        bone_lengths = bone_lengths.copy() + np.random.uniform(-jitter_size, jitter_size, bone_lengths.shape)
        # then make the left and right parts the same lengths

        # the cls.flip_array is joints flip array, but after "- 1" we get the bone flip_array.
        for cur_pair in (cls.flip_array - 1):
            mean_length = (bone_lengths[cur_pair[0]] + bone_lengths[cur_pair[1]]) / 2.0

            bone_lengths[cur_pair[0]] = mean_length
            bone_lengths[cur_pair[1]] = mean_length

        return bone_lengths

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
