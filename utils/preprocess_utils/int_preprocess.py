import numpy as np
import cv2
import os
import sys
import networkx

from common import *
import common

sys.path.append(os.path.dirname(__file__))

class INTProcessor(object):
    def __init__(self, skeleton, img_size):
        self.skeleton = skeleton
        self.bone_indices = self.skeleton.bone_indices
        self.sigma = sigma

        self.flip_array = self.skeleton.flip_array
        self.n_joints = self.skeleton.n_joints
        self.n_bones = self.skeleton.n_bones
        self.img_size = img_size

        self.settings = {
            "img_size": self.img_size,
            "crop_box_size": self.img_size,
            "num_of_joints": self.n_joints,
            "scale_range": 0.15,# max is 0.5 no scale now
            "rotate_range": 30, # max 45
            "shift_range": 0, # pixel
            "is_flip": 1,
            "pad_color": [0.5, 0.5, 0.5],
            "flip_array": self.flip_array
        }

    def preprocess(self, img, joints_2d, is_training=True):
        if np.max(img) > 2:
            img = img / 255.0

        if is_training:
            img[:, :, 0] *= np.random.uniform(0.8, 1.2)
            img[:, :, 1] *= np.random.uniform(0.8, 1.2)
            img[:, :, 2] *= np.random.uniform(0.8, 1.2)
            img = np.clip(img, 0.0, 1.0)

            annots = joints_2d
            aug_img, aug_annot = augment_data_2d(img, annots, self.settings)
            aug_joints_2d = aug_annot[:, 0:2]

        else:
            aug_img = img
            aug_joints_2d = joints_2d

        return aug_img, aug_joints_2d
