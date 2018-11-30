import numpy as np
import sys
import os
import cv2

sys.path.append("../../")

from utils.visualize_utils import display_utils
from utils.defs.skeleton import mSkeleton15 as skeleton

datasets = ["mpii"]

head_scales = dict(lsp=0.8, mpii=0.69)

img_path = lambda x, y: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/{}/images/{}.jpg".format(x, y)

full_labels_dir = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/{}/our_labels".format(x)

full_labels_source = lambda x, y: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/{}/our_labels/{}.txt".format(x, y)
full_labels_target = lambda x, y: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/{}/full_labels/{}.npy".format(x, y)

range_file_path = lambda x: "../../data_range/{}_range.npy".format(x)

def adjust_head(joints_2d, scale):
    head_joint_source = joints_2d[7].copy()
    head_joint_target = joints_2d[8].copy()
    head_length = np.linalg.norm(head_joint_target - head_joint_source)
    head_dir = (head_joint_target - head_joint_source) / head_length

    head_length *= scale

    joints_2d[8] = head_length * head_dir + head_joint_source

    return joints_2d

if __name__ == "__main__":
    for cur_dataset in datasets:
        labels_list = []
        for cur_label in os.listdir(full_labels_dir(cur_dataset)):
            if cur_label.split(".")[1] == "txt":
                labels_list.append(int(cur_label.split(".")[0]))

        labels_list.sort()

        for cur_label in labels_list:
            with open(full_labels_source(cur_dataset, cur_label), "r") as f:
                cur_data = f.readlines()
                cur_2d = np.reshape([float(i) for i in cur_data[0].strip().split(" ")], [-1, 2])
                # adjust the head length
                cur_2d = adjust_head(cur_2d, head_scales[cur_dataset])
                cur_bone_status = np.array([int(i) for i in cur_data[1].strip().split(" ")])
                cur_bone_relations = np.reshape([int(i) for i in cur_data[2].strip().split(" ")], [cur_2d.shape[0]-1, cur_2d.shape[0]-1])

                np.save(full_labels_target(cur_dataset, cur_label), {"joints_2d": cur_2d, "bone_status": cur_bone_status, "bone_relations": cur_bone_relations})

                # cur_img = cv2.imread(img_path(cur_dataset, cur_label))
                # cv2.imshow("test", display_utils.drawLines(cur_img, cur_2d, indices=skeleton.bone_indices, color_table=skeleton.bone_colors*255))
                # cv2.waitKey()

        np.save(range_file_path(cur_dataset), np.array(labels_list))

