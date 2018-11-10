import os
import sys
import numpy as np
import cv2

def get_crop_from_joints(img, joints_2d, crop_box_size = 256, pad_color=[127.5, 127.5, 127.5], pad_scale=0.4):
    img_width = img.shape[1]
    img_height = img.shape[0]

    min_x = np.min(joints_2d[:, 0])
    max_x = np.max(joints_2d[:, 0])
    min_y = np.min(joints_2d[:, 1])
    max_y = np.max(joints_2d[:, 1])

    cen_x = (min_x + max_x) / 2
    cen_y = (min_y + max_y) / 2

    crop_img_size = int(max(max_x - min_x, max_y - min_y) * (1 + pad_scale))

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

def data_resize_with_center_cropped(img, joints_2d, crop_box_size = 256, target_size = 256):
    points = np.reshape(joints_2d, (-1, 2))
    img_cropped, offset_cropped, is_discard = get_crop_from_joints(img, joints_2d, crop_box_size=crop_box_size)

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
