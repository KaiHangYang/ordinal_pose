import numpy as np
import cv2
import os
import sys

from common import *

def preprocess(img, annots, do_rotate=False, is_training=True):
    settings = {
        "img_size": 256,
        "crop_box_size": 256,
        "num_of_joints": 17,
        "scale_range": 0.25,# max is 0.5
        "rotate_range": 30.0 if do_rotate else 0, # max 45
        "shift_range": 0, # pixel
        "is_flip": 1,
        "pad_color": [0.5, 0.5, 0.5],
        "flip_array": np.array([[11, 14], [12, 15], [13, 16], [1, 4], [2, 5], [3, 6]])
    }

    img = img.astype(np.float32)

    # normalize to [0, 1.0]
    if np.max(img) > 2:
        img = img / 255.0

    if is_training:
        # augment the color of the images
        img[:, :, 0] *= np.random.uniform(0.8, 1.2)
        img[:, :, 1] *= np.random.uniform(0.8, 1.2)
        img[:, :, 2] *= np.random.uniform(0.8, 1.2)
        img = np.clip(img, 0.0, 1.0)

        aug_img, aug_annot = augment_data_2d(img, annots, settings)
    else:
        aug_img = img
        aug_annot = annots

    return aug_img, aug_annot

