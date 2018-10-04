import os
import sys
import cv2
import numpy as np

sys.path.append("../")
from common_utils import h36m_camera

# The depths is related to the root joints
def local_to_global(depths, root_depth, joints_2d, source_str, center, scale, crop_box_size=256):
    depths = depths.copy()
    joints_2d = joints_2d.copy()

    joints_3d = np.ones([len(joints_2d), 3])
    subject_num = int(source_str.split("_")[0][1])
    camera_num = int(source_str.split(".")[1].split("_")[0])

    proj_mat, cam_matrix = h36m_camera.get_cam_mat(subject_num, camera_num)
    cur_crop_box_size = crop_box_size * scale
    offset = center - cur_crop_box_size / 2
    joints_2d = joints_2d * scale + offset

    joints_3d[:, 0:2] = joints_2d
    depths += root_depth
    depths = depths[:, np.newaxis]

    joints_3d = depths * np.transpose(np.dot(np.linalg.inv(cam_matrix[0:3, 0:3]), np.transpose(joints_3d)))

    # The proj_mat is used only for visualization
    return joints_2d, joints_3d, proj_mat

# default wnd_width and wnd_height are the size of the human3.6m raw_img
def put_cropped_back(cropped_img, center, scale, crop_box_size=256, wnd_width=1000, wnd_height=1000):
    raw_img = 128 * np.ones([wnd_height, wnd_width, 3], dtype=np.uint8)

    crop_box_size = int(crop_box_size * scale)
    cropped_img = cv2.resize(cropped_img, (crop_box_size, crop_box_size))

    r_l = int(center[0] - crop_box_size / 2)
    r_r = int(r_l + crop_box_size)
    r_t = int(center[1] - crop_box_size / 2)
    r_b = int(r_t + crop_box_size)
    c_l = 0
    c_r = crop_box_size
    c_t = 0
    c_b = crop_box_size

    if r_l < 0:
        c_l = -r_l
        r_l = 0
    if r_r >= wnd_width:
        c_r = crop_box_size - (r_r - wnd_width)
        r_r = wnd_width
    if r_t < 0:
        c_t = -r_t
        r_t = 0
    if r_b >= wnd_height:
        c_b = crop_box_size - (r_b - wnd_height)
        r_b = wnd_width

    c_size = (raw_img[r_t:r_b, r_l:r_r].shape[1], raw_img[r_t:r_b, r_l:r_r].shape[0])
    raw_img[r_t:r_b, r_l:r_r] = cv2.resize(cropped_img[c_t:c_b, c_l:c_r], c_size)

    return raw_img

# Volume size 64*64*17*64
def get_joints_from_volume(volumes, volume_size=64, nJoints=17):
    joints = np.zeros([nJoints, 3], dtype=np.float32)

    for j_idx in range(nJoints):
        cur_vols = volumes[:, :, volume_size*j_idx:volume_size*(j_idx + 1)]
        vol_joints = np.unravel_index(np.argmax(cur_vols), [volume_size, volume_size, volume_size])
        joints[j_idx] = [vol_joints[1], vol_joints[0], vol_joints[2]]

    return joints
