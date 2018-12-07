import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
from network_utils import mConvBnRelu
from network_utils import m_l1_loss
import hourglass

# The structure is translate from github.com/umich-vl/pose-hg-train/blob/maskter/src/models/hg.lua

# is_training is a tensor or python bool
class mDLCMSynNet(object):
    def __init__(self, skeleton, is_training, batch_size, img_size, loss_weights, loss_weight_fb, loss_weight_br, pose_2d_scale, is_use_bn, nFeats=256, nModules=1):
        self.model_name = "DLCMSynNet"

        self.loss_weights = loss_weights
        self.loss_weight_fb = loss_weight_fb
        self.loss_weight_br = loss_weight_br

        self.pose_2d_scale = pose_2d_scale

        self.skeleton = skeleton
        self.nJoints = self.skeleton.n_joints
        self.img_size = img_size
        self.is_use_bias = True
        self.is_tiny = False
        self.is_use_bn = is_use_bn
        self.is_training = is_training
        self.res_utils = mResidualUtils(is_training=self.is_training, is_use_bias=self.is_use_bias, is_tiny=self.is_tiny, is_use_bn=self.is_use_bn)
        self.batch_size = batch_size
        self.feature_size = 64

        self.result_maps = []

        self.nModules = nModules
        self.nRegModules = 2
        self.nParts = self.skeleton.level_nparts
        self.nStacks = self.skeleton.level_n * 2 - 1
        self.nFeats = nFeats
        self.nSemanticLevels = self.skeleton.level_n

    def build_model(self, input_images):
        with tf.variable_scope(self.model_name):
            net = mConvBnRelu(inputs=input_images, nOut=64, kernel_size=3, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="conv1")
            net = tf.layers.max_pooling2d(net, 2, 2, name="pooling1")
            net = mConvBnRelu(inputs=net, nOut=64, kernel_size=3, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="conv2")
            net = mConvBnRelu(inputs=net, nOut=64, kernel_size=3, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="conv3")

            net = self.res_utils.residual_block(net, 128, name="res1")
            net = tf.layers.max_pooling2d(net, 2, 2, name="pooling2")
            net = self.res_utils.residual_block(net, 128, name="res2")
            net = self.res_utils.residual_block(net, self.nFeats, name="res3")

            inter = net

            for l_idx in range(self.nSemanticLevels):
                with tf.variable_scope("b-u_seg_level_{}".format(l_idx)):
                    hg = hourglass.build_hourglass(inter, self.nFeats, 4, name="hg", is_training=self.is_training, nModules=self.nModules, res_utils=self.res_utils)

                    ll = hg
                    ll = mConvBnRelu(inputs=ll, nOut=self.nFeats, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin")

                    n_heatmaps = self.nParts[l_idx]
                    tmp_out = tf.layers.conv2d(inputs=ll, filters=n_heatmaps, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="result_map")

                    self.result_maps.append(tmp_out)
                    ## Add predictions back if this is not the last hg module
                    # if len(self.result_maps) < self.nStacks:
                    with tf.variable_scope("add_block"):
                        ll_ = tf.layers.conv2d(inputs=ll, filters=self.nFeats, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="ll_")
                        tmp_out_ = tf.layers.conv2d(inputs=tmp_out, filters=self.nFeats, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="tmp_out_")
                        inter = tf.add_n([inter, ll_, tmp_out_])

            for l_idx in range(self.nSemanticLevels-1)[::-1]:
                with tf.variable_scope("t-d_seg_level_{}".format(l_idx)):

                    with tf.variable_scope("input_block"):
                        n_remain_channels = self.nFeats - self.nParts[l_idx]
                        inter_ = self.res_utils.residual_block(inter, n_remain_channels, name="res")
                        inter = tf.concat([self.result_maps[l_idx], inter_], axis=3)

                    hg = hourglass.build_hourglass(inter, self.nFeats, 4, name="hg", is_training=self.is_training, nModules=self.nModules, res_utils=self.res_utils)

                    ll = hg
                    ll = mConvBnRelu(inputs=ll, nOut=self.nFeats, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin")

                    n_heatmaps = self.nParts[l_idx]
                    tmp_out = tf.layers.conv2d(inputs=ll, filters=n_heatmaps, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="result_map")

                    self.result_maps.append(tmp_out)

                    ## Add predictions back if this is not the last hg module
                    # if len(self.result_maps) < self.nStacks:
                    with tf.variable_scope("add_block"):
                        ll_ = tf.layers.conv2d(inputs=ll, filters=self.nFeats, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="ll_")
                        tmp_out_ = tf.layers.conv2d(inputs=tmp_out, filters=self.nFeats, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="tmp_out_")
                        inter = tf.add_n([inter, ll_, tmp_out_])

            reg = inter
            with tf.variable_scope("relation_blocks"):
                for i in range(4):
                    with tf.variable_scope("final_downsample_{}".format(i)):
                        for j in range(self.nRegModules):
                            reg = self.res_utils.residual_block(reg, self.nFeats, name="res{}".format(j))
                        reg = tf.layers.max_pooling2d(reg, pool_size=2, strides=2, padding="VALID", name="maxpool")

                with tf.variable_scope("final_classify"):
                    reg = tf.layers.flatten(reg)
                    # fb information and br matrix (half matrix)
                    self.relations = tf.layers.dense(inputs=reg, units= 3 * self.nJoints * (self.nJoints - 1) / 2, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc")
                    self.relations = tf.reshape(self.relations, [self.batch_size, -1, 3])

            assert(len(self.result_maps) == self.nStacks)

    # input_joints shape (None, 17, 2)
    def build_input_heatmaps(self, input_center, stddev=2.0, name="input_heatmaps", gaussian_coefficient=False):
        with tf.variable_scope(name):
            raw_arr_y = tf.constant(np.reshape(np.repeat(np.arange(0, self.feature_size, 1), self.feature_size), [self.feature_size, self.feature_size, 1]).astype(np.float32), name="raw_arr_y")

            const_y = tf.tile(raw_arr_y[np.newaxis], [self.batch_size, 1, 1, 1])
            const_x = tf.tile(tf.transpose(raw_arr_y, perm=[1, 0, 2])[np.newaxis], [self.batch_size, 1, 1, 1])

            all_heatmaps = []
            for j_idx in range(self.nJoints):

                if gaussian_coefficient:
                    cur_heatmaps = (1.0 / (2 * np.pi * stddev * stddev)) * tf.exp(-(tf.pow(const_x - tf.reshape(input_center[:, j_idx, 0], [-1, 1, 1, 1]), 2) + tf.pow(const_y - tf.reshape(input_center[:, j_idx, 1], [-1, 1, 1, 1]), 2)) / 2.0 / stddev / stddev)
                else:
                    cur_heatmaps = tf.exp(-(tf.pow(const_x - tf.reshape(input_center[:, j_idx, 0], [-1, 1, 1, 1]), 2) + tf.pow(const_y - tf.reshape(input_center[:, j_idx, 1], [-1, 1, 1, 1]), 2)) / 2.0 / stddev / stddev)

                all_heatmaps.append(cur_heatmaps)

            heatmaps_labels = tf.concat(all_heatmaps, axis=3, name="heatmaps")

        return heatmaps_labels

    def get_joints_hm(self, heatmaps, batch_size, name="heatmap_to_joints"):
        with tf.variable_scope(name):
            with tf.device("/device:GPU:0"):
                max_indices = tf.argmax(tf.reshape(heatmaps, [batch_size, -1, self.nJoints]), axis=1)

            # currently the unravel_index only support cpu.
            with tf.device("cpu:0"):
                cur_joints = tf.unravel_index(tf.reshape(max_indices, [-1]), [self.feature_size, self.feature_size])
            cur_joints = tf.reshape(tf.transpose(cur_joints), [-1, self.nJoints, 2])[:, :, ::-1]

        return tf.cast(cur_joints, tf.float32)

    def cal_accuracy(self, gt_joints, pd_joints, name="accuracy"):
        with tf.variable_scope(name):
            accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(gt_joints - pd_joints, 2), axis=2)))
        return accuracy

    def flip_heatmaps(self, hms, flip_array, name="flip_heatmaps"):
        with tf.variable_scope(name):
            # flip the features map first
            flipped_hms = tf.image.flip_left_right(hms)
            # get the new hm orders
            old_order = np.arange(0, self.nJoints, 1)
            cur_order = old_order.copy()

            cur_order[flip_array[:, 0]] = old_order[flip_array[:, 1]]
            cur_order[flip_array[:, 1]] = old_order[flip_array[:, 0]]

            cur_hms = []
            for j in range(self.nJoints):
                cur_idx = cur_order[j]
                cur_hms.append(flipped_hms[:, :, :, cur_idx][:, :, :, tf.newaxis])

            return tf.concat(cur_hms, axis=3, name="flipped_hms")

    def build_evaluation(self, flip_array):
        # here I assume the input image has raw_image and flipped_image
        with tf.variable_scope("extract_heatmap"):
            cur_batch_size = tf.cast(self.batch_size, dtype=tf.int32) / 2

            raw_heatmaps = self.result_maps[-1][0:cur_batch_size]
            flipped_heatmaps = self.flip_heatmaps(self.result_maps[-1][cur_batch_size:], flip_array, name="flip_heatmaps")

            combined_heatmaps = tf.concat([raw_heatmaps, flipped_heatmaps], axis=0)

            all_pd_2d = self.get_joints_hm(combined_heatmaps, batch_size=2*cur_batch_size, name="heatmap_to_joints") * self.pose_2d_scale

            self.raw_pd_2d = all_pd_2d[0:cur_batch_size]
            self.mean_pd_2d = (all_pd_2d[0:cur_batch_size] + all_pd_2d[cur_batch_size:]) / 2

        self.fb_info = self.results[0:cur_batch_size, 0:self.nJoints-1]
        # 1 is in front, 0 is unconcerned or unknown, -1 is behind
        self.br_info = self.results[0:cur_batch_size, self.nJoints-1:]

        with tf.variable_scope("extract_fb"):
            self.pd_fb_result = tf.argmax(self.fb_info, axis=2)
            self.pd_fb_belief = tf.reduce_max(self.fb_info, axis=2)

        with tf.variable_scope("extract_br"):
            self.pd_br_result = tf.argmax(self.br_info, axis=2)
            self.pd_br_belief = tf.reduce_max(self.br_info, axis=2)

    def build_loss(self, input_maps, input_fb, input_br, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        self.total_loss = 0
        with tf.variable_scope("losses"):

            # 1 is forward, 0 is uncertain, -1 is backward
            self.fb_info = self.relations[:, 0:self.nJoints-1]
            # 1 is in front, 0 is unconcerned or unknown, -1 is behind
            self.br_info = self.relations[:, self.nJoints-1:]

            self.fb_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_fb, logits=self.fb_info, dim=-1, name="fb_loss")) / self.batch_size * self.loss_weight_fb
            self.br_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_br, logits=self.br_info, dim=-1, name="br_loss")) / self.batch_size * self.loss_weight_br

            self.total_loss = self.total_loss + self.fb_loss + self.br_loss

            self.losses = []
            for pd_idx, gt_idx in enumerate(range(self.nSemanticLevels) + range(self.nSemanticLevels-1)[::-1]):
                tmp_loss = tf.nn.l2_loss(input_maps[gt_idx] - self.result_maps[pd_idx], name="heatmaps_loss_{}".format(pd_idx)) / self.batch_size * self.loss_weights[gt_idx]
                self.losses.append(tmp_loss)
                self.total_loss = self.total_loss + tmp_loss


        # NOTICE: The dependencies must be added, because of the BN used in the residual 
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Update_ops num {}".format(len(update_ops)))
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.total_loss, self.global_steps)

        with tf.variable_scope("accuracy"):
            ######### heatmap accuracy
            with tf.variable_scope("heatmap_acc"):
                cur_batch_size = tf.cast(self.batch_size, dtype=tf.int32)
                combined_heatmaps = tf.concat([input_maps[0], self.result_maps[-1]], axis=0)
                all_joints_2d = self.get_joints_hm(combined_heatmaps, batch_size=2*cur_batch_size, name="heatmap_to_joints")

                self.gt_2d = all_joints_2d[0:cur_batch_size] * self.pose_2d_scale
                self.pd_2d = all_joints_2d[cur_batch_size:] * self.pose_2d_scale

                self.heatmaps_acc = self.cal_accuracy(gt_joints=self.gt_2d, pd_joints=self.pd_2d, name="joints_2d_acc")

            with tf.variable_scope("fb_accuracy"):
                self.pd_fb_result = tf.argmax(self.fb_info, axis=2)
                self.gt_fb_result = tf.argmax(input_fb, axis=2)

                self.pd_fb_belief = tf.reduce_max(self.fb_info, axis=2)
                self.gt_fb_belief = tf.reduce_max(input_fb, axis=2)

                self.fb_acc = tf.reduce_mean(tf.cast(tf.equal(self.pd_fb_result, self.gt_fb_result), dtype=tf.float32))

            with tf.variable_scope("br_accuracy"):
                self.pd_br_result = tf.argmax(self.br_info, axis=2)
                self.gt_br_result = tf.argmax(input_br, axis=2)

                self.pd_br_belief = tf.reduce_max(self.br_info, axis=2)
                self.gt_br_belief = tf.reduce_max(input_br, axis=2)

                self.br_acc = tf.reduce_mean(tf.cast(tf.equal(self.pd_br_result, self.gt_br_result), dtype=tf.float32))


        tf.summary.scalar("fb_loss_scalar", self.fb_loss)
        tf.summary.scalar("br_loss_scalar", self.br_loss)
        tf.summary.scalar("fb_acc_scalar", self.fb_acc)
        tf.summary.scalar("br_acc_scalar", self.br_acc)

        tf.summary.scalar("total_loss_scalar", self.total_loss)
        for i in range(self.nStacks):
            tf.summary.scalar("component_heatmaps_loss_{}".format(i), self.losses[i])
        tf.summary.scalar("heatmaps_acc_scalar", self.heatmaps_acc)

        tf.summary.scalar("learning_rate", self.lr)
        self.merged_summary = tf.summary.merge_all()
