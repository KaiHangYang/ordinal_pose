import os
import sys
import numpy as np

sys.path.append("../")
from utils.postprocess_utils import skeleton_opt
from utils.postprocess_utils import volume_utils
from utils.defs import pose_defs

labels_dir = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/train/labels_syn"
labels_path_fn = lambda x: labels_dir + "/{}.npy".format(x)

if __name__ == "__main__":
    n_labels = len(os.listdir(labels_dir))

    for i in range(n_labels):
        cur_label = np.load(labels_path_fn(i)).tolist()

        label_scale = cur_label["scale"]
        label_offset = cur_label["center"] - 128 * label_scale
        label_joints_2d = cur_label["joints_2d"]
        label_joints_3d = cur_label["joints_3d"]
        label_cam_mat = cur_label["cam_mat"]

        cur_depths = label_joints_3d[:, 2].copy()[:, np.newaxis]
        root_depth = cur_depths[0]
        cur_depths = cur_depths - root_depth

        ############## The way use the ground truth depht and (scale, center) #############
        # cur_3d = np.ones([17, 3])
        # cur_3d[:, 0:2] = label_joints_2d * label_scale + label_offset
        # cam_matrix = label_cam_mat.copy()

        # gtd_joints_3d = (cur_depths + root_depth) * np.transpose(np.dot(np.linalg.inv(cam_matrix[0:3, 0:3]), np.transpose(cur_3d)))
        # print(np.max(np.abs(gtd_joints_3d - label_joints_3d)))
        ###################################################################################

        ############### The way use the mean skeleton ##############
        opt_joints_3d = np.reshape(skeleton_opt.opt(label_joints_2d.flatten().tolist(), cur_depths.flatten().tolist()), [-1, 3])
        opt_joints_3d = opt_joints_3d - opt_joints_3d[0]
        label_joints_3d = label_joints_3d - label_joints_3d[0]
        print(np.mean(np.sqrt(np.sum((opt_joints_3d - label_joints_3d) ** 2, axis=1))))
        ############################################################
