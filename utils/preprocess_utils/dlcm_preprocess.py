import numpy as np
import cv2
import os
import sys
import networkx

from common import *
import common

sys.path.append(os.path.dirname(__file__))

class SynProcessor(object):
    def __init__(self, skeleton, img_size, hm_size):
        self.joint_ratio=joint_ratio
        self.bg_color=bg_color
        self.skeleton = skeleton
        self.bone_indices = self.skeleton.bone_indices

        self.flip_array = self.skeleton.flip_array
        self.n_joints = self.skeleton.n_joints
        self.n_bones = self.skeleton.n_bones
        self.img_size = img_size
        self.hm_size = hm_size

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
        if p1[0] == p2[0] and p1[1] == p2[1]:
            return self.draw_gaussain(img, p1, sigma)
        segment_joints = self.get_segment_points(p1, p2)

        for cur_joint in segment_joints:
            img = self.draw_gaussain(img, cur_joint, sigma)

    def compose_maps(self, img, children, idx_set):
    

    def preprocess_base(self, img, joints_2d, bone_status, bone_relations, is_training=True):
        if np.max(img) > 2:
            img = img / 255.0

        if is_training:
            img[:, :, 0] *= np.random.uniform(0.8, 1.2)
            img[:, :, 1] *= np.random.uniform(0.8, 1.2)
            img[:, :, 2] *= np.random.uniform(0.8, 1.2)
            img = np.clip(img, 0.0, 1.0)

            array_order = np.arange(0, self.n_joints, 1) - 1

            annots = np.concatenate([joints_2d, array_order[:, np.newaxis]], axis=1)
            aug_img, aug_annot = augment_data_2d(img, annots, self.settings)

            aug_joints_2d = aug_annot[:, 0:2]

            aug_array_order = aug_annot[:, 2][1:].astype(np.int32)
            array_order = array_order[1:].astype(np.int32)

            aug_bone_relations = np.zeros_like(bone_relations)
            aug_bone_status = np.zeros_like(bone_status)

            for i in range(self.n_bones):
                aug_bone_status[aug_array_order[i]] = bone_status[array_order[i]]
                for j in range(self.n_bones):
                    aug_bone_relations[aug_array_order[i]][aug_array_order[j]] = bone_relations[array_order[i]][array_order[j]]

        else:
            aug_img = img
            aug_joints_2d = joints_2d
            aug_bone_relations = bone_relations
            aug_bone_status = bone_status

        return aug_img, aug_joints_2d, aug_bone_status, aug_bone_relations

    # TODO the joints_2d must be the cropped one, the joints_3d shouldn't be the root related
    def preprocess_h36m(self, img, joints_2d, joints_3d, scale, center, cam_mat, is_training=True):
        joints_2d = joints_2d.copy()
        joints_3d = joints_3d.copy()

        raw_joints_2d = joints_2d.copy()
        raw_joints_3d = joints_3d.copy()
        ############ Calculate the bone_relations and bone_status ############
        bone_order, bone_relations = self.get_bone_relations(raw_joints_2d, raw_joints_3d, scale, center, cam_mat)
        bone_status = self.recalculate_bone_status(raw_joints_3d[:, 2])
        ######################################################################

        return self.preprocess_base(img, joints_2d, bone_status, bone_relations, is_training)
