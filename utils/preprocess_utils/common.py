import sys
import numpy as np
import cv2
import random
import math
import time


############################### Preprocess #####################################3
## Settings for the augment_data_2d


'''
 Functional: Make the gaussian heatmap
 Parameter:  point: the gaussian center
             size: the heatmap size
             ratio: the gaussian ratio
'''
def make_gaussian(point, size=46, ratio=3):
    if point[0] <= 0.000001 and point[1] <= 0.000001:
        return np.zeros((size, size), np.float32)

    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = point[0]
    y0 = point[1]

    heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / ratio/ ratio)
    return heatmap

'''
 Functional: Augmentate the 2d data
 Parameter:  img: the img to augment
             annots: the annots
             settings: the augment settings(include: img_size, crop_box_size, num_of_joints)
'''

def augment_data_2d(img, annots, settings = {
        "img_size": 256,
        "crop_box_size": 256,
        "num_of_joints": 17,
        "scale_range": 0.0,# max is 0.5
        "rotate_range": 0.0, # max 45
        # "shift_range": 0.0,
        "is_flip": 0,
        "pad_color": [128, 128, 128],
        "flip_array": None
    }):

    # Extract the settings 
    img_size = settings["img_size"]
    crop_box_size = settings["crop_box_size"]
    num_of_joints = settings["num_of_joints"]
    scale_range = settings["scale_range"]
    rotate_range = settings["rotate_range"]
    # shift_range = settings["shift_range"]
    is_flip = settings["is_flip"]
    pad_color = settings["pad_color"]
    flip_array = settings["flip_array"] # contain the joints pairs which need to be exchanged if do flip

    scale_size = (-1 if random.random() >= 0.5 else 1) * random.random() * scale_range
    rotate_size = (-1 if random.random() >= 0.5 else 1) * random.random() * rotate_range

    # shift_size_l = (-1 if random.random() >= 0.5 else 1) * random.random() * shift_range
    # shift_size_t = (-1 if random.random() >= 0.5 else 1) * random.random() * shift_range

    do_flip = (0 if random.random() >= 0.5 else 1) * is_flip

    if do_flip:
        assert(flip_array is not None)
    # scale first
    cur_scale = 1 + scale_size
    scaled_img = cv2.resize(img, (int(img_size * cur_scale), int(img_size * cur_scale)))
    offset_size = int((scaled_img.shape[0] - crop_box_size) / 2.0)

    # pad the image to 2*img_size 
    if scaled_img.shape[0] < 2*img_size and scaled_img.shape[1] < 2*img_size:
        pad_y = (2*img_size - scaled_img.shape[0]) / 2
        pad_x = (2*img_size - scaled_img.shape[1]) / 2
        padded_img = cv2.copyMakeBorder(scaled_img, top=pad_y, bottom=pad_y, left=pad_x, right=pad_x, value=pad_color, borderType=cv2.BORDER_CONSTANT)

    # rotate by the img center
    rotate_mat = cv2.getRotationMatrix2D((padded_img.shape[1] / 2, padded_img.shape[0] / 2), rotate_size, 1.0)
    rotated_img = cv2.warpAffine(padded_img, rotate_mat, (int(padded_img.shape[1]), int(padded_img.shape[0])), borderMode=cv2.BORDER_CONSTANT, borderValue=pad_color)

    # the crop
    # crop_center = np.array([rotated_img.shape[1] / 2.0 + shift_size_l, rotated_img.shape[0] / 2.0 + shift_size_t]).astype(np.int32)
    crop_center = np.array([rotated_img.shape[1] / 2.0, rotated_img.shape[0] / 2.0]).astype(np.int32)

    offset_l = crop_center[0] - crop_box_size/2
    offset_t = crop_center[1] - crop_box_size/2

    cropped_img = rotated_img[offset_t:offset_t + crop_box_size, offset_l:offset_l+crop_box_size]

    pt_offset_l = img_size * scale_size / 2.0
    pt_offset_t = img_size * scale_size / 2.0

    if do_flip:
        result_img = cv2.flip(cropped_img, 1)
        # print("Image flipped!")
    else:
        result_img = cropped_img

    # Then do the transform to all the points
    for c_num in range(num_of_joints):
        cur_p = annots[c_num][0:2]
        if cur_p.any():
            cur_p *= cur_scale
            cur_p = np.dot(rotate_mat, np.array([cur_p[0], cur_p[1], 1]))
            cur_p -= np.array([pt_offset_l, pt_offset_t])

            if (cur_p > 0).all() and (cur_p < crop_box_size).all():
                if do_flip:
                    annots[c_num][0:2] = np.array([crop_box_size - 1 - cur_p[0], cur_p[1]])
                else:
                    annots[c_num][0:2] = cur_p
            else:
                annots[c_num][0:2] = 0
    if do_flip:
        for flip_pair in flip_array:
            tmp_annot = annots[flip_pair[0]].copy()
            annots[flip_pair[0]] = annots[flip_pair[1]].copy()
            annots[flip_pair[1]] = tmp_annot

    return result_img, annots

def img2show(train_image, image_pixel_range):
    result_img = ((train_image - image_pixel_range[0]) / (image_pixel_range[1] - image_pixel_range[0])) * 255
    return result_img.astype(np.uint8)

def img2train(img, image_pixel_range):
    return (image_pixel_range[1] - image_pixel_range[0]) * (img / 255.0) + image_pixel_range[0]

def get_crop_from_center(img, center, scale, crop_box_size = 256, pad_color=[128, 128, 128]):
    img_width = img.shape[1]
    img_height = img.shape[0]

    cen_x = center[0]
    cen_y = center[1]

    crop_img_size = int(crop_box_size * scale)

    min_x = int(cen_x - crop_img_size / 2)
    min_y = int(cen_y - crop_img_size / 2)
    max_x = min_x + crop_img_size
    max_y = min_y + crop_img_size

    raw_min_x = 0
    raw_max_x = img_width
    raw_min_y = 0
    raw_max_y = img_height

    if min_x > 0:
        raw_min_x = min_x
    if min_y > 0:
        raw_min_y = min_y
    if max_x < img_width:
        raw_max_x = max_x
    if max_y < img_height:
        raw_max_y = max_y

    is_discard = False
    cropped_img = cv2.copyMakeBorder(img[raw_min_y:raw_max_y, raw_min_x:raw_max_x], top=raw_min_y-min_y, bottom=max_y - raw_max_y, left=raw_min_x - min_x, right=max_x - raw_max_x, value=pad_color, borderType=cv2.BORDER_CONSTANT)
    if cropped_img is None or cropped_img.shape[0] != cropped_img.shape[1]:
        is_discard = True
    return cropped_img, (min_x, min_y, max_x, max_y), is_discard

def data_resize_with_center_cropped(img, joints2d, center, scale, crop_box_size = 256, target_size = 256, num_of_joints=17):
    points = np.reshape(joints2d, (-1, 2))
    img_cropped, offset_cropped, is_discard = get_crop_from_center(img, center, scale, crop_box_size=crop_box_size)

    if is_discard:
        return img, points, offset_cropped, 1.0, is_discard

    for i in range(points.shape[0]):
        if not (points[i, 0] == 0 and points[i, 1] == 0):
            points[i, 0] = points[i, 0] - offset_cropped[0]
            points[i, 1] = points[i, 1] - offset_cropped[1]

    scale = float(img_cropped.shape[0]) / target_size

    img_result = cv2.resize(img_cropped, (int(target_size), int(target_size)))

    points = np.float32(points)
    points /= scale

    return img_result, points, is_discard

def get_relation_table(joints_z):
    relation_table = np.zeros([len(joints_z), len(joints_z)], dtype=np.int32)

    for i in range(len(joints_z)):
        for j in range(i+1, len(joints_z)):
            if np.abs(joints_z[i] - joints_z[j]) < 100:
                # i is closer than j
                relation_table[i, j] = 0
                relation_table[j, i] = 0
            elif joints_z[i] < joints_z[j]:
                relation_table[i, j] = 1
                relation_table[j, i] = -1
            else:
                relation_table[i, j] = -1
                relation_table[j, i] = 1

    return relation_table

