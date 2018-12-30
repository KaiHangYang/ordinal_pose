import numpy as np
import cv2
import os
import sys
import networkx
import math

from common import *
import common

sys.path.append(os.path.dirname(__file__))

import get_bone_relations as gbr_module

class PoseProcessor(object):
    ## The pure_color parameter means the joints only have color (black, gray, white)
    def __init__(self, skeleton, img_size, with_br, with_fb=True, bone_width=6, joint_ratio=6, overlap_threshold=6, bone_status_threshold=80, bg_color=0.2, pad_scale=0.4, angle_jitter_size=math.pi/20, bonelength_jitter_size=30, pure_color=True):
        self.with_br = with_br
        self.with_fb = with_fb
        self.pure_color = pure_color

        self.pad_scale = pad_scale
        self.angle_jitter_size = angle_jitter_size
        self.bonelength_jitter_size = bonelength_jitter_size

        self.bone_width=bone_width
        self.joint_ratio=joint_ratio
        self.overlap_threshold = overlap_threshold
        self.bone_status_threshold = bone_status_threshold
        self.bg_color=bg_color
        self.skeleton = skeleton
        self.bone_indices = self.skeleton.bone_indices
        self.flip_array = self.skeleton.flip_array
        self.bone_colors = self.skeleton.bone_colors
        self.n_joints = self.skeleton.n_joints
        self.n_bones = self.skeleton.n_bones
        self.img_size = img_size

        self.settings = {
            "img_size": self.img_size,
            "crop_box_size": self.img_size,
            "num_of_joints": self.n_joints,
            "scale_range": 0.00,# max is 0.5 no scale now
            "rotate_range": 0, # max 45
            "shift_range": 0, # pixel
            "is_flip": 0,
            "pad_color": [0, 0, 0],
            "flip_array": self.flip_array
        }

    def get_bone_relations(self, joints_2d, joints_3d, scale, center, cam_mat):
        joints_2d = joints_2d.flatten().tolist()
        joints_3d = joints_3d.flatten().tolist()
        scale = scale
        center = list(center)
        cam_vec = [cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2]]
        skeleton_type = self.skeleton.skeleton_index

        ################ Debug ###############
        # Track the pose with bug
        ######################################
        # np.save("/home/kaihang/Projects/pose_project/new_pose/train/train_log/bug_pose.npy", {"joints_2d": joints_2d, "joints_3d":joints_3d, "scale": scale, "center": center, "cam_vec": cam_vec, "img_size": self.img_size, "overlap_threshold": self.overlap_threshold})
        result_data = gbr_module.get_bone_relations(joints_2d, joints_3d, scale, center, cam_vec, self.img_size, self.overlap_threshold, skeleton_type)

        result_data = np.reshape(result_data, [-1, self.skeleton.n_bones])

        bone_order = result_data[0]
        bone_relations = result_data[1:]

        return bone_order, bone_relations

    def assemble_pose(self, angles, bone_lengths, root_pos, cam_mat):
        raw_joints_3d = self.skeleton.get_joints(angles, bone_lengths) + root_pos

        raw_joints_2d = np.transpose(np.dot(cam_mat, np.transpose(raw_joints_3d)))
        raw_joints_2d = (raw_joints_2d / raw_joints_2d[:, 2][:, np.newaxis])[:, 0:2]
        joints_2d, center, scale = common.crop_joints_2d(raw_joints_2d, target_box_size=self.img_size, pad_scale=self.pad_scale)

        return joints_2d, raw_joints_3d, center, scale


    def preprocess(self, angles, bone_lengths, root_pos, cam_mat, is_training=True):

        if is_training:
            angles = self.skeleton.jitter_angles(angles, jitter_size=self.angle_jitter_size)
            bone_lengths = self.skeleton.jitter_bonelengths(bone_lengths, jitter_size=self.bonelength_jitter_size)

        joints_2d, joints_3d, center, scale = self.assemble_pose(angles=angles, bone_lengths=bone_lengths, root_pos=root_pos, cam_mat=cam_mat)

        #### Paint the synthetic image
        if self.with_br:
            bone_order, _ = self.get_bone_relations(joints_2d, joints_3d, scale, center, cam_mat)
        else:
            bone_order = np.arange(0, self.n_bones, 1)

        if self.with_fb:
            bone_status = self.recalculate_bone_status(joints_3d[:, 2])
        else:
            bone_status = np.zeros([self.n_bones])

        img = self.draw_syn_img(joints_2d=joints_2d, bone_status=bone_status, bone_order=bone_order)

        if np.max(img) >= 2:
            img = img / 255.0

        ####  Set the joints_3d related to the root
        joints_3d = joints_3d - joints_3d[0]

        return img, joints_2d, joints_3d

    def recalculate_bone_status(self, joints_z):
        bone_status = []
        for cur_bone_idx in self.bone_indices:
            if np.abs(joints_z[cur_bone_idx[0]] - joints_z[cur_bone_idx[1]]) < self.bone_status_threshold:
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

            if self.pure_color:
                if bone_status[cur_bone] == 0:
                    # not certain
                    cur_joint_color = (127, 127, 127)
                elif bone_status[cur_bone] == 1:
                    # forward
                    cur_joint_color = (255, 255, 255)
                else:
                    # backward
                    cur_joint_color = (0, 0, 0)
            else:
                if bone_status[cur_bone] == 0:
                    # not certain
                    cur_joint_color = (cur_bone_color * 0.5).astype(cur_bone_color.dtype)
                elif bone_status[cur_bone] == 1:
                    # forward
                    cur_joint_color = (cur_bone_color * 0.8).astype(cur_bone_color.dtype)
                else:
                    # backward
                    cur_joint_color = (cur_bone_color * 0.2).astype(cur_bone_color.dtype)


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
