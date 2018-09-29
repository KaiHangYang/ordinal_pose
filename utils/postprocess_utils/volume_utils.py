import os
import sys
import cv2
import numpy as np

sys.path.append("../")
from common_utils import h36m_camera

def depth_to_joints_3d(depths, root_depth, joints_2d, cur_label, crop_box_size=256):
    subject_num = int(cur_label["source"].split("_")[0][1])
    camera_num = int(cur_label["source"].split(".")[1].split("_")[0])

    proj_mat, _ = h36m_camera.get_cam_mat(subject_num, camera_num)
    center = cur_label["center"]
    scale = cur_label["scale"]
    cur_crop_box_size = crop_box_size * scale
    offset = center - cur_crop_box_size / 2
    joints_2d = joints_2d * scale + offset

    depths += root_depth
