import numpy as np
import cv2
import os
import sys

from common import *

def preprocess(img, annots):
    settings = {
        "img_size": 256,
        "crop_box_size": 256,
        "num_of_joints": 17,
        "scale_range": 0.25,# max is 0.5
        "rotate_range": 30, # max 45
        # "shift_range": 0.0,
        "is_flip": 1,
        "pad_color": [128, 128, 128],
        "flip_array": np.array([[11, 14], [12, 15], [13, 16], [1, 4], [2, 5], [3, 6]])
    }

    aug_img, aug_annot = augment_data_2d(img, annots, settings)

    return aug_img, aug_annot

