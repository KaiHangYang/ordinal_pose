import numpy as np
import cv2
import os
import sys
import networkx

from common import *
import common

flip_array = np.array([[1, 4], [2, 5], [3, 6], [7, 10], [8, 11], [9, 12]])

bg_img_dir = "/home/kaihang/DataSet_2/Ordinal/human3.6m/cropped_256/bg_images"
bg_img_arr = None

if not os.path.exists(bg_img_dir):
    print("The background image directory doesn't exist!")
    exit(-1)


def replace_bg(img, mask):
    global bg_img_arr
    global bg_img_dir

    # assume the img size is constant in the dataset
    # random select the bg first
    img_width = img.shape[1]
    img_height = img.shape[0]

    if bg_img_arr is None:
        # initialize the bg_img_arr once
        bg_img_arr = []
        bg_list = os.listdir(bg_img_dir)

        for bg_path in bg_list:
            cur_bg_img = cv2.imread(os.path.join(bg_img_dir, bg_path))
            min_img_size = min(img_width, img_height)
            min_bg_size = min(cur_bg_img.shape[1], cur_bg_img.shape[0])

            if min_bg_size < min_img_size:
                scale = float(min_img_size) / min_bg_size
                cur_bg_img = cv2.resize(cur_bg_img, (0, 0), fx=scale, fy=scale)

            bg_img_arr.append(cur_bg_img)

    np.random.shuffle(bg_img_arr)

    ###### NOTICE: Here I promise I won't change the bg img ######
    bg_img = bg_img_arr[0]

    bg_width = bg_img.shape[1]
    bg_height = bg_img.shape[0]

    # random crop the bg
    offset_l_max = bg_width - img_width
    offset_t_max = bg_height - img_height

    offset_l = int(np.random.random() * offset_l_max)
    offset_t = int(np.random.random() * offset_t_max)

    cur_mask = (mask > 50).all(axis=2)

    img[cur_mask] = bg_img[offset_t:offset_t+img_height, offset_l:offset_l+img_width][cur_mask]

    return img

# currently I don't augment the data
# data_type(0) means the human3.6m data, data_type(1) means the lsp and mpii data.

def preprocess(img, joints_2d, bone_status, is_training=True, mask=None):

    settings = {
        "img_size": 256,
        "crop_box_size": 256,
        "num_of_joints": 13,
        "scale_range": 0.1,# max is 0.5 no scale now
        "rotate_range": 10.0, # max 45
        "shift_range": 0, # pixel
        "is_flip": 1,
        "pad_color": [0.5, 0.5, 0.5],
        "flip_array": flip_array
    }

    # first change the background
    if mask is not None:
        # roll to determine whether changing the background
        if np.random.uniform() >= 0.5:
            # random change the background from the background image directory
            img = replace_bg(img, mask)

    img = img.astype(np.float32).copy()

    # normalize to [0, 1.0]
    if np.max(img) > 2:
        img = img / 255.0

    if is_training:
        # augment the color of the images
        img[:, :, 0] *= np.random.uniform(0.8, 1.2)
        img[:, :, 1] *= np.random.uniform(0.8, 1.2)
        img[:, :, 2] *= np.random.uniform(0.8, 1.2)
        img = np.clip(img, 0.0, 1.0)

        annots = np.concatenate([joints_2d, np.concatenate([[-1], bone_status])[:, np.newaxis]], axis=1)
        aug_img, aug_annot = augment_data_2d(img, annots, settings)

        aug_joints_2d = aug_annot[:, 0:2]
        aug_bone_status = aug_annot[:, 2][1:].astype(np.int32)

    else:
        aug_img = img
        aug_joints_2d = joints_2d
        aug_bone_status = bone_status

    return aug_img, aug_joints_2d, aug_bone_status

def flip_annots(joints_2d, bone_status, size=256):

    annots = np.concatenate([joints_2d, np.concatenate([[-1], bone_status])[:, np.newaxis]], axis=1)
    flipped_annots = common._flip_annot(annots, flip_array, size=size)

    flipped_joints_2d = flipped_annots[:, 0:2]
    flipped_bone_status = flipped_annots[:, 2][1:].astype(np.int32)

    return flipped_joints_2d, flipped_bone_status
