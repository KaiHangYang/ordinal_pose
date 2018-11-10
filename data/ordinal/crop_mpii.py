import numpy as np
import os
import sys
import cv2
import scipy.io as sio

sys.path.append("../../")
from utils.visualize_utils import display_utils
import crop_utils

mpii_data_dir = "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/raw_data/mpii_upis1h"

source_img_path = lambda x: os.path.join(mpii_data_dir, "images/{:05d}_full.jpg").format(x+1)
source_2d_path = os.path.join(mpii_data_dir, "joints.mat")
source_ord_path = os.path.join(mpii_data_dir, "ordinal.mat")

target_img_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/mpii/images/{}.jpg".format(x)
target_lbl_path = lambda x: "/home/kaihang/DataSet_2/Ordinal/lsp_mpii/cropped_256/mpii/labels/{}.npy".format(x)

new_index = np.array([6, 3, 4, 5, 2, 1, 0, 8, 9, 13, 14, 15, 12, 11, 10]).astype(np.int32)

mpii_bone_indices = np.array([
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

raw_mpii_bone_indicies = np.array([
    [2, 3], # 0
    [3, 4], # 1
    [4, 5], # 2
    [3, 2], # 3
    [2, 1], # 4
    [1, 0], # 5
    [6, 6], # 6 the spin need to be determind by the two hip joint and throat joint
    [8, 9], # 7
    [8, 13], # 8
    [13, 14], # 9
    [14, 15], # 10
    [8, 12], # 11
    [12, 11], # 12
    [11, 10], # 13
    ])

if __name__ == "__main__":
    # total 2000 images, from 1 - 2000
    img_range = np.arange(0, 13030, 1)
    lbl_2d = np.transpose(sio.loadmat(source_2d_path)["joints"], axes=[2, 1, 0])
    lbl_ord = sio.loadmat(source_ord_path)["ord"]
    for idx in img_range:
        sys.stderr.write("\rCurrently process {}".format(idx))
        sys.stderr.flush()

        cur_img = cv2.imread(source_img_path(idx))
        cur_2d = lbl_2d[idx][:, 0:2][new_index]

        #add the root joint by getting the mean of the two hip joints
        cur_bone_status = []
        cur_joint_color = [[255, 255, 255]]
        cur_ord = lbl_ord[idx]

        for cur_bone in raw_mpii_bone_indicies:
            cur_bone_status.append(cur_ord[cur_bone[0]][cur_bone[1]])

        # determind the spin bone status
        if cur_ord[8][2] == 1 and cur_ord[8][3] == 1:
            cur_bone_status[6] = -1
        elif cur_ord[8][2] == -1 and cur_ord[8][3] == -1:
            cur_bone_status[6] = 1
        else:
            cur_bone_status[6] = 0

        for t_idx, t_status in enumerate(cur_bone_status):
            if t_status == 1:
                # forward
                t_color = [255, 255, 255]
                cur_bone_status[t_idx] = 1
            elif t_status == 0:
                # uncertain
                t_color = [128, 128, 128]
                cur_bone_status[t_idx] = 0
            else:
                # backward
                t_color = [0, 0, 0]
                cur_bone_status[t_idx] = 2

            cur_joint_color.append(t_color)

        cur_img, cur_2d, _ = crop_utils.data_resize_with_center_cropped(cur_img, cur_2d, crop_box_size=256, target_size=256)

        cv2.imwrite(target_img_path(idx), cur_img)
        np.save(target_lbl_path(idx), {"joints_2d": cur_2d, "bone_status": cur_bone_status})

        # cur_img = display_utils.drawLines(cur_img, cur_2d, mpii_bone_indices)
        # cur_img = display_utils.drawPoints(cur_img, cur_2d, text_scale=0.4, point_color_table=cur_joint_color, point_ratio=5)
        # cv2.imshow("test", cur_img)
        # cv2.waitKey()
