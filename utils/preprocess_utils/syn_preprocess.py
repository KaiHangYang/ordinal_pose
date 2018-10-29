import numpy as np
import cv2
import os
import sys

from common import *
import common

flip_array = np.array([[11, 14], [12, 15], [13, 16], [1, 4], [2, 5], [3, 6]])

# currently I don't augment the data
def preprocess(img, joints_2d, bone_status, bone_order, is_training=True):
    settings = {
        "img_size": 256,
        "crop_box_size": 256,
        "num_of_joints": 17,
        "scale_range": 0.3,# max is 0.5 no scale now
        "rotate_range": 30.0, # max 45
        "shift_range": 0, # pixel
        "is_flip": 1,
        "pad_color": [0.5, 0.5, 0.5],
        "flip_array": flip_array
    }

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

        bone_order = bone_order.astype(np.int32)
        order_arr = np.zeros_like(bone_order)
        for i in range(len(bone_order)):
            order_arr[bone_order[i]] = i

        annots = np.concatenate([joints_2d, np.concatenate([[-1], bone_status])[:, np.newaxis], np.concatenate([[-1], order_arr])[:, np.newaxis]], axis=1)
        aug_img, aug_annot = augment_data_2d(img, annots, settings)

        aug_joints_2d = aug_annot[:, 0:2]
        aug_bone_status = aug_annot[:, 2][1:].astype(np.int32)

        aug_order_arr = aug_annot[:, 3][1:].astype(np.int32)

        aug_bone_order = np.zeros_like(bone_order)
        for i in range(len(aug_order_arr)):
            aug_bone_order[aug_order_arr[i]] = i

    else:
        aug_img = img

        aug_joints_2d = joints_2d
        aug_bone_status = bone_status
        aug_bone_order = bone_order

    return aug_img, aug_joints_2d, aug_bone_status, aug_bone_order

def flip_data(img, annots, size=64):
    return common._flip_data(img, annots, flip_array, size)

def flip_annot(annots, size=64):
    return common._flip_annot(annots, flip_array, size=size)

# draw the ground truth joints_2d
bones_indices = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [8, 11],
        [11, 12],
        [12, 13],
        [8, 14],
        [14, 15],
        [15, 16]
])


bone_colors = np.array([
    [1.000000, 1.000000, 0.000000],
    [0.492543, 0.000000, 0.000000],
    [0.000000, 1.000000, 0.000000],
    [1.000000, 0.349454, 0.000000],
    [0.499439, 0.558884, 1.000000],
    [0.000000, 0.362774, 0.000000],
    [0.500312, 0.000000, 0.624406],
    [0.501744, 0.724322, 0.275356],
    [0.000000, 1.000000, 1.000000],
    [1.000000, 0.000000, 1.000000],
    [1.000000, 0.499433, 0.611793],
    [1.000000, 0.800000, 1.000000],
    [0.000000, 0.502502, 0.611632],
    [0.200000, 0.700000, 0.300000],
    [0.700000, 0.300000, 0.100000],
    [0.300000, 0.200000, 0.800000]
])

def draw_syn_img(joints_2d, bone_status, bone_order, size=64, bg_color=0.2, bone_width=1, joint_ratio=1):

    # assert(joints_2d.max() < size)

    bone_sum = bone_status.shape[0]
    bg_color = int(bg_color * 255)

    sep_synmaps = np.array([bg_color * np.ones([size, size, 3]) for _ in range(bone_sum)]).astype(np.uint8)
    synmap = bg_color * np.ones([size, size, 3]).astype(np.uint8)

    for cur_bone in bone_order:
        ####### get bone informations first #######
        cur_bone_color = np.array([bone_colors[cur_bone][2] * 255, bone_colors[cur_bone][1] * 255, bone_colors[cur_bone][0] * 255])

        if bone_status[cur_bone] == 0:
            # not certain
            cur_joint_color = (127, 127, 127)
        elif bone_status[cur_bone] == 1:
            # forward
            cur_joint_color = (255, 255, 255)
        else:
            # backward
            cur_joint_color = (0, 0, 0)

        source_joint = joints_2d[bones_indices[cur_bone][0]]
        target_joint = joints_2d[bones_indices[cur_bone][1]]
        ###########################################

        dir_2d = (target_joint - source_joint) / (0.0000001 + np.linalg.norm(target_joint - source_joint))
        w_vec = np.array([dir_2d[1], -dir_2d[0]])

        joint_to_draw = tuple(np.round(target_joint).astype(np.int32))

        vertex_1 = source_joint - bone_width / 2.0 * w_vec;
        vertex_2 = source_joint + bone_width / 2.0 * w_vec;

        vertex_3 = target_joint + bone_width / 2.0 * w_vec;
        vertex_4 = target_joint - bone_width / 2.0 * w_vec;

        vertices = np.array([[np.round(vertex_1[0]), np.round(vertex_1[1])],
                             [np.round(vertex_2[0]), np.round(vertex_2[1])],
                             [np.round(vertex_3[0]), np.round(vertex_3[1])],
                             [np.round(vertex_4[0]), np.round(vertex_4[1])]]).astype(np.int32)

        synmap = cv2.fillConvexPoly(synmap, vertices, cur_bone_color, cv2.LINE_AA);
        synmap = cv2.circle(synmap, joint_to_draw, joint_ratio, cur_joint_color, cv2.FILLED, cv2.LINE_AA);

        sep_synmaps[cur_bone] = cv2.fillConvexPoly(sep_synmaps[cur_bone], vertices, cur_bone_color, cv2.LINE_AA);
        sep_synmaps[cur_bone] = cv2.circle(sep_synmaps[cur_bone], joint_to_draw, joint_ratio, cur_joint_color, cv2.FILLED, cv2.LINE_AA);

    return synmap, sep_synmaps
