import numpy as np
import os
import sys
import cv2
import scipy.io as sio

sys.path.append("../../")
from utils.visualize_utils import display_utils

lsp_data_dir = "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/raw_data/lsp_dataset_original"

source_img_path = lambda x: os.path.join(lsp_data_dir, "images/im{:04d}.jpg").format(x+1)
source_2d_path = os.path.join(lsp_data_dir, "joints.mat")
source_ord_path = os.path.join(lsp_data_dir, "ordinal.mat")

target_img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/lsp/images/{}.jpg"
target_lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/lsp/labels/{}.npy"

new_index = np.array([3, 4, 5, 2, 1, 0, 12, 13, 9, 10, 11, 8, 7, 6]).astype(np.int32)

lsp_bone_indices = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [0, 4],
    [4, 5],
    [5, 6],
    [0, 7],
    [7, 8],
    [7, 9],
    [9, 10],
    [10, 11],
    [7, 12],
    [12, 13],
    [13, 14]
    ], dtype=np.int32)

raw_lsp_bone_indicies = np.array([
    [2, 3], # 0
    [3, 4], # 1
    [4, 5], # 2
    [3, 2], # 3
    [2, 1], # 4
    [1, 0], # 5
    [12, 12], # 6 the spin bone need to be determined by the two hip joints
    [12, 13], # 7
    [12, 9], # 8
    [9, 10], # 9
    [10, 11], # 10
    [12, 8], # 11
    [8, 7], # 12
    [7, 6], # 13
    ])

if __name__ == "__main__":
    # total 2000 images, from 1 - 2000
    img_range = np.arange(0, 2000, 1)

    lbl_2d = np.transpose(sio.loadmat(source_2d_path)["joints"], axes=[2, 1, 0])
    lbl_ord = sio.loadmat(source_ord_path)["ord"]

    for idx in img_range:
        cur_img = cv2.imread(source_img_path(idx))
        cur_2d = lbl_2d[idx][:, 0:2][new_index]
        # add the root joint by getting the mean of the two hip joints
        cur_2d = np.concatenate([[(cur_2d[0] + cur_2d[3]) / 2], cur_2d], axis=0)
        cur_bone_status = []
        cur_joint_color = [[255, 255, 255]]
        cur_ord = lbl_ord[idx]

        for cur_bone in raw_lsp_bone_indicies:
            cur_bone_status.append(cur_ord[cur_bone[0]][cur_bone[1]])

        # determind the spin bone status
        if cur_ord[12][2] == 1 and cur_ord[12][3] == 1:
            cur_bone_status[6] = -1
        elif cur_ord[12][2] == 1 and cur_ord[12][3] == 1:
            cur_bone_status[6] = -1

        for t_status in cur_bone_status:
            if t_status == 1:
                t_color = [255, 255, 255]
            elif t_status == 0:
                t_color = [128, 128, 128]
            else:
                t_color = [0, 0, 0]

            cur_joint_color.append(t_color)

        cur_img = display_utils.drawLines(cur_img, cur_2d, lsp_bone_indices)
        cur_img = display_utils.drawPoints(cur_img, cur_2d, text_scale=0.4, point_color_table=cur_joint_color)
        cv2.imshow("test", cur_img)
        cv2.waitKey()
