import numpy as np
import os
import sys

sys.path.append("../../../")

from utils.defs.skeleton import mSkeleton15 as skeleton

train_or_valid = "valid"

source_label_dir = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/"+train_or_valid+"/labels_syn/"
target_label_dir = "/home/kaihang/DataSet_2/Ordinal/syn/"+train_or_valid

if __name__ == "__main__":

    source_lbl_list = []

    # select only the npy file
    for cur_lbl in os.listdir(source_label_dir):
        if cur_lbl.split(".")[1] == "npy":
            source_lbl_list.append(cur_lbl)

    for idx, cur_lbl in enumerate(source_lbl_list):
        sys.stdout.write("\rCurrent Processing: {}".format(idx))
        sys.stdout.flush()
        # Now only use the skeleton-15
        lbl_data = np.load(os.path.join(source_label_dir, cur_lbl)).tolist()
        joints_3d = lbl_data["joints_3d"].copy()[skeleton.h36m_selected_index]
        joints_2d = lbl_data["joints_2d"].copy()[skeleton.h36m_selected_index]

        root_pos = joints_3d[0].copy()
        bone_lengths = skeleton.get_bonelengths(joints_3d)
        angles = skeleton.get_angles(joints_3d)

        new_lbl = {
            "cam_mat": lbl_data["cam_mat"][0:3, 0:3],
            "scale": lbl_data["scale"],
            "center": lbl_data["center"],
            "joints_3d": joints_3d,
            "joints_2d": joints_2d,
            "root_pos": root_pos,
            "bone_lengths": bone_lengths,
            "angles": angles
        }

        np.save(os.path.join(target_label_dir, cur_lbl), new_lbl)
