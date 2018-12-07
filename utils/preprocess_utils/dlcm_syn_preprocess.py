import numpy as np
import cv2
import os
import sys
import networkx

from common import *
import common

sys.path.append(os.path.dirname(__file__))

import get_bone_relations as gbr_module

class DLCMSynProcessor(object):
    def __init__(self, skeleton, img_size, hm_size, sigma=1.0, bone_width=6, joint_ratio=6, bg_color=0.2):
        self.bone_width=bone_width
        self.joint_ratio=joint_ratio
        self.bg_color=bg_color
        self.skeleton = skeleton
        self.bone_indices = self.skeleton.bone_indices
        self.flip_array = self.skeleton.flip_array
        self.bone_colors = self.skeleton.bone_colors
        self.n_joints = self.skeleton.n_joints
        self.n_bones = self.skeleton.n_bones
        self.img_size = img_size
        self.hm_size = hm_size
        self.sigma = sigma
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

    ###################### Functions to draw the heatmaps #########################
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

        return img

    def compose_maps(self, img, children, idx_set):
        for i in idx_set:
            img = np.maximum(img, children[:, :, i])
        return img
    #############################################################################3

    def get_bone_relations(self, joints_2d, joints_3d, scale, center, cam_mat):
        joints_2d = joints_2d.flatten().tolist()
        joints_3d = joints_3d.flatten().tolist()
        scale = scale
        center = center.tolist()
        cam_vec = [cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2]]
        skeleton_type = self.skeleton.skeleton_index

        result_data = gbr_module.get_bone_relations(joints_2d, joints_3d, scale, center, cam_vec, self.img_size, 2*self.joint_ratio, skeleton_type)
        result_data = np.reshape(result_data, [-1, self.skeleton.n_bones])

        bone_order = result_data[0]
        bone_relations = result_data[1:]

        return bone_order, bone_relations

    def flip_annots(self, joints_2d, bone_status, _bone_relations, _bone_relations_belief):
        # the bone_relations is the flattened array
        array_order = np.arange(0, self.n_joints, 1) - 1

        annots = np.concatenate([joints_2d, array_order[:, np.newaxis]], axis=1)
        flipped_annots = common._flip_annot(annots, self.flip_array, size=self.img_size)

        flipped_joints_2d = flipped_annots[:, 0:2]
        flipped_array_order = flipped_annots[:, 2][1:].astype(np.int32)
        array_order = array_order[1:].astype(np.int32)

        bone_relations = np.zeros([self.n_bones, self.n_bones])
        bone_relations_belief = np.zeros([self.n_bones, self.n_bones])
        bone_relations[np.triu_indices(self.n_bones, k=1)] = _bone_relations
        bone_relations_belief[np.triu_indices(self.n_bones, k=1)] = _bone_relations_belief

        flipped_bone_relations = np.zeros([self.n_bones, self.n_bones])
        flipped_bone_relations_belief = np.zeros([self.n_bones, self.n_bones])

        flipped_bone_status = np.zeros_like(bone_status)

        for i in range(self.n_bones):
            flipped_bone_status[flipped_array_order[i]] = bone_status[array_order[i]]
            for j in range(self.n_bones):
                flipped_bone_relations[flipped_array_order[i]][flipped_array_order[j]] = bone_relations[array_order[i]][array_order[j]]
                flipped_bone_relations_belief[flipped_array_order[i]][flipped_array_order[j]] = bone_relations_belief[array_order[i]][array_order[j]]

        return flipped_joints_2d, flipped_bone_status, flipped_bone_relations[np.triu_indices(self.n_bones, k=1)], flipped_bone_relations_belief[np.triu_indices(self.n_bones, k=1)]

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

        ############################ Make the datamaps #################################
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

        return aug_img, result_maps, aug_joints_2d, aug_bone_status, aug_bone_relations

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

    def recalculate_bone_status(self, joints_z):
        bone_status = []
        for cur_bone_idx in self.bone_indices:
            if np.abs(joints_z[cur_bone_idx[0]] - joints_z[cur_bone_idx[1]]) < 100:
                bone_status.append(0)
            elif joints_z[cur_bone_idx[1]] < joints_z[cur_bone_idx[0]]:
                bone_status.append(1)
            else:
                bone_status.append(2)
        return np.array(bone_status)

    # the bone_relations is a up triangle matrix
    def bone_order_from_bone_relations(self, bone_relations, bone_relations_belief):
        br_mat = np.zeros([self.n_bones, self.n_bones])
        brb_mat = np.zeros([self.n_bones, self.n_bones])

        if bone_relations.shape[0] == self.n_bones:
            br_mat = bone_relations
            brb_mat = bone_relations_belief
        else:
            br_mat[np.triu_indices(self.n_bones, k=1)] = bone_relations
            brb_mat[np.triu_indices(self.n_bones, k=1)] = bone_relations_belief

        br_graph = networkx.DiGraph()
        # initialize the graph with all the nodes
        br_graph.add_nodes_from(np.arange(0, self.n_bones, 1).astype(np.int32))
        bone_order = []

        for i in range(self.n_bones):
            for j in range(i+1, self.n_bones):
                if br_mat[i][j] == 1:
                    # i is in front of j
                    br_graph.add_weighted_edges_from([(j, i, brb_mat[i][j])])
                elif br_mat[i][j] == 2:
                    # i is behind j
                    br_graph.add_weighted_edges_from([(i, j, brb_mat[i][j])])

        # get the render order from the graph
        while len(bone_order) != self.n_bones:
            is_selected_one = False
            for cur_bone_idx, cur_degree in br_graph.in_degree_iter():
                if cur_degree == 0:
                    is_selected_one = True
                    bone_order.append(cur_bone_idx)
                    br_graph.remove_node(cur_bone_idx)

            if not is_selected_one:
                # there is a cycle, then find the cycle and delete the minimum-weighted edge
                one_cycle = networkx.simple_cycles(br_graph).next()

                min_weight = float("inf")
                for idx in range(-1, len(one_cycle)-1, 1):
                    source = one_cycle[idx]
                    target = one_cycle[idx+1]

                    if min_weight > br_graph.edge[source][target]["weight"]:
                        min_weight = br_graph.edge[source][target]["weight"]
                        edge_to_delete = [source, target]

                br_graph.remove_edge(edge_to_delete[0], edge_to_delete[1])

        return bone_order

    def draw_syn_img(self, joints_2d, bone_status, bone_order):
        # assert(joints_2d.max() < size)
        bone_sum = bone_status.shape[0]
        bg_color = int(self.bg_color * 255)

        synmap = bg_color * np.ones([self.img_size, self.img_size, 3]).astype(np.uint8)

        for cur_bone in bone_order:
            ####### get bone informations first #######
            cur_bone_color = np.array([self.bone_colors[cur_bone][2] * 255, self.bone_colors[cur_bone][1] * 255, self.bone_colors[cur_bone][0] * 255])

            if bone_status[cur_bone] == 0:
                # not certain
                cur_joint_color = (127, 127, 127)
            elif bone_status[cur_bone] == 1:
                # forward
                cur_joint_color = (255, 255, 255)
            else:
                # backward
                cur_joint_color = (0, 0, 0)

            source_joint = joints_2d[self.bone_indices[cur_bone][0]]
            target_joint = joints_2d[self.bone_indices[cur_bone][1]]
            ###########################################

            dir_2d = (target_joint - source_joint) / (0.0000001 + np.linalg.norm(target_joint - source_joint))

            w_vec = np.array([dir_2d[1], -dir_2d[0]])

            joint_to_draw = tuple(np.round(target_joint).astype(np.int32))

            vertex_1 = source_joint - self.bone_width / 2.0 * w_vec;
            vertex_2 = source_joint + self.bone_width / 2.0 * w_vec;

            vertex_3 = target_joint + self.bone_width / 2.0 * w_vec;
            vertex_4 = target_joint - self.bone_width / 2.0 * w_vec;


            vertices = np.array([[np.round(vertex_1[0]), np.round(vertex_1[1])],
                                 [np.round(vertex_2[0]), np.round(vertex_2[1])],
                                 [np.round(vertex_3[0]), np.round(vertex_3[1])],
                                 [np.round(vertex_4[0]), np.round(vertex_4[1])]]).astype(np.int32)

            synmap = cv2.fillConvexPoly(synmap, vertices, cur_bone_color, cv2.LINE_AA);
            synmap = cv2.circle(synmap, joint_to_draw, self.joint_ratio, cur_joint_color, cv2.FILLED, cv2.LINE_AA);

        return synmap
