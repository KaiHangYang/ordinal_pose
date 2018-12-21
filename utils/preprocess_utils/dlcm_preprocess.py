import numpy as np
import cv2
import os
import sys
import networkx

from common import *
import common

sys.path.append(os.path.dirname(__file__))

class DLCMProcessor(object):
    def __init__(self, skeleton, img_size, hm_size, sigma=1.0, is_norm=False):
        self.skeleton = skeleton
        self.bone_indices = self.skeleton.bone_indices
        self.sigma = sigma
        self.is_norm = is_norm

        self.flip_array = self.skeleton.flip_array
        self.n_joints = self.skeleton.n_joints
        self.n_bones = self.skeleton.n_bones
        self.img_size = img_size
        self.hm_size = hm_size

        self.joints_2d_scale = float(self.hm_size) / self.img_size

        self.settings = {
            "img_size": self.img_size,
            "crop_box_size": self.img_size,
            "num_of_joints": self.n_joints,
            "scale_range": 0.25,# max is 0.5 no scale now
            "rotate_range": 30, # max 45
            "shift_range": 0, # pixel
            "is_flip": 1,
            "pad_color": [0.5, 0.5, 0.5],
            "flip_array": self.flip_array
        }

    # the pt must be in the heatmaps range [64, 64]
    def draw_gaussain(self, img, p, sigma):
        tmp_gaussain = common.make_gaussian(p, size=self.hm_size, sigma=sigma)

        if self.is_norm:
            tmp_sum = tmp_gaussain.sum()
            if tmp_sum > 0:
                tmp_gaussain = tmp_gaussain / tmp_sum

        img = np.maximum(img, tmp_gaussain)
        return img

    def get_segment_points(self, p1, p2):
        # Bresenham's line algorithm
        # Return all the points located on line segment between (x1, y1) and (x2, y2)
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        steep = np.abs(y2 - y1) > np.abs(x2 - x1)

        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dx = x2 - x1
        dy = np.abs(y2 - y1)

        error = dx / 2.0
        ystep = 1 if y1 < y2 else -1
        y = np.floor(y1)
        maxX = np.floor(x2)
        x = x1

        res = np.zeros([int(maxX-x1), 2])

        for i in range(len(res)):
            if steep:
                res[i][0] = y
                res[i][1] = x
            else:
                res[i][0] = x
                res[i][1] = y

            error = error - dy

            if error < 0:
                y = y + ystep
                error = error + dx
            x += 1

        return res

    def draw_limbmap(self, img, p1, p2, sigma):
        # from 'Deeply Learned Compositional Models for Human Pose Estimation'
        # Draw gaussian heat maps along the limb between p1 and p2
        # p1 and p2 are 1-based locations [x, y]

        if (p1<=0).any() or (p2<=0).any() or (p1>=self.hm_size).any() or (p2>=self.hm_size).any():
            return img

        p1 = np.round(p1)
        p2 = np.round(p2)

        if p1[0] == p2[0] and p1[1] == p2[1]:
            return self.draw_gaussain(img, p1, sigma)

        segment_joints = self.get_segment_points(p1, p2)
        for cur_joint in segment_joints:
            img = self.draw_gaussain(img, cur_joint, sigma)

        return img

    def compose_maps(self, img, children, idx_set):
        for i in idx_set:
            img = np.maximum(img, children[:, :, i])
        return img

    # only use for multiscale evaluation
    def preprocess_multiscale(self, img, joints_2d, scale_range, size=256, pad_color=[128, 128, 128]):
        multiscale_data = []
        for cur_scale in scale_range:
            scaled_img = cv2.resize(img, dsize=None, fx=1+cur_scale, fy=1+cur_scale)
            scaled_joints_2d = joints_2d * (1 + cur_scale)

            ### Then pad and scale
            scaled_img, scaled_joints_2d = common.center_pad_or_crop(scaled_img, scaled_joints_2d, size=size, pad_color=pad_color)
            multiscale_data.append([scaled_img, scaled_joints_2d])

        return multiscale_data

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

        # make the joints_2d within heatmap range
        aug_joints_2d *= self.joints_2d_scale

        # then get the result_maps
        result_maps = []
        ######## first heatmaps level 0
        heatmaps_level_0 = np.zeros([self.hm_size, self.hm_size, self.skeleton.level_nparts[0]])
        for cur_part in self.skeleton.level_structure[0]:
            heatmaps_level_0[:, :, cur_part] = self.draw_gaussain(heatmaps_level_0[:, :, cur_part], p=aug_joints_2d[cur_part], sigma=self.sigma)
        result_maps.append(heatmaps_level_0)
        ######## Second heatmaps level 1
        heatmaps_level_1 = np.zeros([self.hm_size, self.hm_size, self.skeleton.level_nparts[1]])
        for cur_idx, cur_part in enumerate(self.skeleton.level_structure[1]):
            heatmaps_level_1[:, :, cur_idx] = self.draw_limbmap(heatmaps_level_1[:, :, cur_idx], p1=aug_joints_2d[cur_part[0]], p2=aug_joints_2d[cur_part[1]], sigma=self.sigma)
        result_maps.append(heatmaps_level_1)
        ######## the last maps
        for cur_level in range(2, self.skeleton.level_n):
            heatmaps_level_n = np.zeros([self.hm_size, self.hm_size, self.skeleton.level_nparts[cur_level]])
            for cur_idx, cur_part in enumerate(self.skeleton.level_structure[cur_level]):
                heatmaps_level_n[:, :, cur_idx] = self.compose_maps(heatmaps_level_n[:, :, cur_idx], result_maps[cur_level-1], cur_part)
            result_maps.append(heatmaps_level_n)

        return aug_img, result_maps, aug_joints_2d
