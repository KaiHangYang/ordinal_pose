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
             sigma: the gaussian sigma
'''
def make_gaussian(point, size=64, sigma=2):
    if point[0] <= 0 or point[1] <= 0 or point[0] >= size or point[1] >= size:
        return np.zeros((size, size), np.float32)

    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = point[0]
    y0 = point[1]

    heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / sigma/ sigma)
    return heatmap

### Pay attentaion, the function is only used under (n, h, w, c) mode
def make_gaussian_3d(point, size=64, sigma=2):
    if point[0] < 0.000000 or point[1] < 0.000000 or point[2] < 0.000000 or point[0] >= size or point[1] >= size or point[2] >= size:
        return np.zeros((size, size, size), np.float32)

    z = np.arange(0, size, 1, np.float32)
    x = z[:, np.newaxis]
    y = x[:, np.newaxis]

    x0 = point[0]
    y0 = point[1]
    z0 = point[2]

    heatmaps_3d = np.exp(-((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / 2.0 / sigma /sigma)
    return heatmaps_3d
'''
 Functional: Augmentate the multiscale data
 Parameter: img: the input image
            joints_2d: the joints_2d
            size: the target size
'''

def center_pad_or_crop(img, joints_2d, size=256, pad_color=[128, 128, 128]):
    img = img.copy()
    joints_2d = joints_2d.copy()

    img_width = img.shape[1]
    img_height = img.shape[0]
    assert(img_width == img_height)
    offset = (size - img_width) / 2.0

    if offset > 0:
        # !!!!! np.round 0.5 = 0
        if round(offset) > offset:
            offset = int(offset)
            extra_pixel = 1
        else:
            offset = int(offset)
            extra_pixel = 0

        valid_mask = (joints_2d != [0, 0]).all(axis=1)
        result_joints_2d = joints_2d.copy()
        result_joints_2d[valid_mask] = joints_2d[valid_mask] + np.array([offset, offset])
        # pad the image
        result_img = cv2.copyMakeBorder(img, top=offset, left=offset, right=offset+extra_pixel, bottom=offset+extra_pixel, value=pad_color, borderType=cv2.BORDER_CONSTANT)
    else:
        offset = int(offset)

        valid_mask = (joints_2d != [0, 0]).all(axis=1)
        result_joints_2d = joints_2d.copy()
        result_joints_2d[valid_mask] = joints_2d[valid_mask] + np.array([offset, offset])

        offset = -offset
        result_img = img[offset:offset+size, offset:offset+size].copy()

    return result_img, result_joints_2d

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
        "shift_range": 0.0,
        "is_flip": 0,
        "pad_color": [127.5, 127.5, 127.5],
        "flip_array": None
    }):

    # Extract the settings 
    img_size = settings["img_size"]
    crop_box_size = settings["crop_box_size"]
    num_of_joints = settings["num_of_joints"]
    assert(num_of_joints == len(annots))

    scale_range = settings["scale_range"]
    rotate_range = settings["rotate_range"]
    shift_range = settings["shift_range"]
    is_flip = settings["is_flip"]
    pad_color = settings["pad_color"]
    flip_array = settings["flip_array"] # contain the joints pairs which need to be exchanged if do flip

    if isinstance(scale_range, list):
        scale_size = np.random.uniform(scale_range[0], scale_range[1])
    else:
        scale_size = (-1 if random.uniform(0, 1) >= 0.5 else 1) * random.random() * scale_range

    rotate_size = (0 if random.uniform(0, 1) >= 0.4 else 1) * (-1 if random.uniform(0, 1) >= 0.5 else 1) * random.random() * rotate_range

    shift_size_l = (-1 if random.uniform(0, 1) >= 0.5 else 1) * random.random() * shift_range
    shift_size_t = (-1 if random.uniform(0, 1) >= 0.5 else 1) * random.random() * shift_range

    do_flip = (0 if random.uniform(0, 1) >= 0.5 else 1) * is_flip

    # print("scale_size: {}\n. rotate_size: {}\n do_flip: {}".format(scale_size, rotate_size, do_flip))

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
    img_rotate_mat = cv2.getRotationMatrix2D((padded_img.shape[1] / 2, padded_img.shape[0] / 2), rotate_size, 1.0)
    joints_rotate_mat = cv2.getRotationMatrix2D((crop_box_size / 2, crop_box_size / 2), rotate_size, 1.0)
    rotated_img = cv2.warpAffine(padded_img, img_rotate_mat, (int(padded_img.shape[1]), int(padded_img.shape[0])), borderMode=cv2.BORDER_CONSTANT, borderValue=pad_color)

    # the crop
    crop_center = np.array([rotated_img.shape[1] / 2.0 + shift_size_l, rotated_img.shape[0] / 2.0 + shift_size_t]).astype(np.int32)
    # crop_center = np.array([rotated_img.shape[1] / 2.0, rotated_img.shape[0] / 2.0]).astype(np.int32)

    offset_l = crop_center[0] - crop_box_size/2
    offset_t = crop_center[1] - crop_box_size/2

    cropped_img = rotated_img[offset_t:offset_t + crop_box_size, offset_l:offset_l+crop_box_size]

    pt_offset_l = offset_size
    pt_offset_t = offset_size

    pt_shift_l = -shift_size_l
    pt_shift_t = -shift_size_t

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
            cur_p -= np.array([pt_offset_l, pt_offset_t])
            cur_p = np.dot(joints_rotate_mat, np.array([cur_p[0], cur_p[1], 1]))
            cur_p += np.array([pt_shift_l, pt_shift_t])

            if do_flip:
                annots[c_num][0:2] = np.array([crop_box_size - 1 - cur_p[0], cur_p[1]])
            else:
                annots[c_num][0:2] = cur_p

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


def crop_joints_2d(joints_2d, target_box_size=256, pad_scale=0.2):
    min_x = np.min(joints_2d[:, 0])
    min_y = np.min(joints_2d[:, 1])
    max_x = np.max(joints_2d[:, 0])
    max_y = np.max(joints_2d[:, 1])

    center = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0]
    cur_box_size = np.max(max_x - min_x, max_y - min_y) * (1 + pad_scale)

    offset_x = center[0] - cur_box_size / 2.0
    offset_y = center[1] - cur_box_size / 2.0

    scale = cur_box_size / target_box_size

    joints_2d = (joints_2d - [offset_x, offset_y]) / scale

    return joints_2d, center, scale

def get_crop_from_center(img, center, scale, crop_box_size = 256, pad_color=[127.5, 127.5, 127.5]):
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
    relation_table = np.zeros([len(joints_z), len(joints_z)], dtype=np.float32)

    loss_table_log = np.zeros([len(joints_z), len(joints_z)], dtype=np.float32)
    loss_table_pow = np.zeros([len(joints_z), len(joints_z)], dtype=np.float32)

    for i in range(len(joints_z)):
        for j in range(i+1, len(joints_z)):
            if np.abs(joints_z[i] - joints_z[j]) < 100:
                # i is closer than j
                relation_table[i, j] = 0

                loss_table_log[i, j] = 0
                loss_table_pow[i, j] = 1

            elif joints_z[i] < joints_z[j]:
                relation_table[i, j] = 1

                loss_table_log[i, j] = 1
                loss_table_pow[i, j] = 0

            else:
                relation_table[i, j] = -1
                loss_table_log[i, j] = 1
                loss_table_pow[i, j] = 0

    return relation_table, loss_table_log, loss_table_pow

def _flip_data(img, annots, flip_array, size):
    flipped_img = img.copy()
    flipped_annots = annots.copy()

    flipped_img = cv2.flip(flipped_img, 1)
    flipped_annots = _flip_annot(flipped_annots, flip_array, size)

    return flipped_img, flipped_annots

def flip_img(img):
    return cv2.flip(img.copy(), 1)

def _flip_annot(annots, flip_array, size):
    flipped_annots = annots.copy()

    for a_num in range(flipped_annots.shape[0]):
        cur_annot = flipped_annots[a_num][0:2]
        flipped_annots[a_num][0:2] = np.array([size - 1 - cur_annot[0], cur_annot[1]])

    for flip_pair in flip_array:
        tmp_annot = flipped_annots[flip_pair[0]].copy()
        flipped_annots[flip_pair[0]] = flipped_annots[flip_pair[1]].copy()
        flipped_annots[flip_pair[1]] = tmp_annot

    return flipped_annots

