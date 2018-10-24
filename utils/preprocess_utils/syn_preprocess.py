import numpy as np
import cv2
import os
import sys

from common import *
import common

flip_array = np.array([[11, 14], [12, 15], [13, 16], [1, 4], [2, 5], [3, 6]])

# currently I don't augment the data
# def preprocess(img, annots, is_training=True, is_rotate=True):
def preprocess(img, syn_img, annots):
    # settings = {
        # "img_size": 256,
        # "crop_box_size": 256,
        # "num_of_joints": 17,
        # "scale_range": 0.25,# max is 0.5
        # "rotate_range": 30.0 if is_rotate else 0.0, # max 45
        # "shift_range": 0, # pixel
        # "is_flip": 1,
        # "pad_color": [0.2, 0.2, 0.2],
        # "flip_array": flip_array
    # }

    img = img.astype(np.float32).copy()
    syn_img = syn_img.astype(np.float32).copy()

    # normalize to [0, 1.0]
    if np.max(img) > 2:
        img = img / 255.0

    if np.max(syn_img) > 2:
        syn_img = syn_img / 255.0

    # if is_training:
        # # augment the color of the images
        # img[:, :, 0] *= np.random.uniform(0.8, 1.2)
        # img[:, :, 1] *= np.random.uniform(0.8, 1.2)
        # img[:, :, 2] *= np.random.uniform(0.8, 1.2)
        # img = np.clip(img, 0.0, 1.0)

        # pad_size = img.shape[0] / 2

        # aug_img, aug_annot = augment_data_2d(img, annots, settings)
    # else:
        # aug_img = img
        # aug_annot = annots

    # return aug_img, aug_annot
    return img, syn_img, annots

def flip_data(img, annots, size=64):
    return common._flip_data(img, annots, flip_array, size)

def flip_annot(annots, size=64):
    return common._flip_annot(annots, flip_array, size=size)
