import numpy as np
import os
import sys

sys.path.append("../../../")

from utils.defs.skeleton import mSkeleton15 as skeleton

train_or_valid = "train"

source_label_dir = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+train_or_valid+"/labels_syn/"
target_label_dir = "/home/kaihang/DataSet_2/Ordinal/syn/"+train_or_valid

if __name__ == "__main__":

    source_lbl_list = []

    # select only the npy file
    for cur_lbl in os.listdir(source_label_dir):
        if cur_lbl.split(".")[1] == "npy":
            source_lbl_list.append(cur_lbl)

    root_pos_array = []
    bone_lengths_array = []
    angle_array = []
    cam_mat_array = []

    for idx, cur_lbl in enumerate(source_lbl_list):
        sys.stdout.write("\rCurrent Processing: {}".format(idx))
        sys.stdout.flush()
        # Now only use the skeleton-15
        lbl_data = np.load(os.path.join(source_label_dir, cur_lbl)).tolist()
        joints_3d = lbl_data["joints_3d"].copy()[skeleton.h36m_selected_index]
        joints_2d = lbl_data["joints_2d"].copy()[skeleton.h36m_selected_index]
        cam_mat = lbl_data["cam_mat"]

        root_pos = joints_3d[0].copy()
        bone_lengths = skeleton.get_bonelengths(joints_3d)
        angles = skeleton.get_angles(joints_3d)

        cam_mat_array.append(cam_mat)
        bone_lengths_array.append(bone_lengths)
        angle_array.append(angles)
        root_pos_array.append(root_pos)

    np.save(os.path.join(target_label_dir, "angles.npy"), angle_array)
    np.save(os.path.join(target_label_dir, "bone_lengths.npy"), bone_lengths_array)
    np.save(os.path.join(target_label_dir, "root_pos.npy"), root_pos_array)
    np.save(os.path.join(target_label_dir, "cam_mat.npy"), cam_mat_array)

