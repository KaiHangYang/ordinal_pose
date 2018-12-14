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
class mINTNet(object):
    def __init__(self, skeleton, is_training, batch_size, img_size, loss_weight_heatmap, loss_weight_integral, pose_2d_scale, is_use_bn, nFeats=256, nModules=1, nStacks=1):
        self.model_name = "Net_Int"

        self.loss_weight_heatmap = loss_weight_heatmap
        self.loss_weight_integral = loss_weight_integral
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

        self.heatmaps = []
        self.integrals = []

        self.nModules = nModules
        self.nStacks = nStacks
        self.nFeats = nFeats

        with tf.variable_scope("heat_vec"):
            self.heat_vec = tf.constant(np.arange(0, self.feature_size, 1).astype(np.float32)[np.newaxis, :, np.newaxis])
            self.heat_vec = tf.tile(self.heat_vec, [self.batch_size, 1, self.nJoints])

    def build_integral(self, input_heatmap, name="integral"):
        with tf.variable_scope(name):
            hm_softmax = tf.reshape(tf.nn.softmax(tf.reshape(input_heatmap, [self.batch_size, -1, self.nJoints]), axis=1), [self.batch_size, self.feature_size, self.feature_size, self.nJoints])

            hm_x_vec = tf.reduce_sum(hm_softmax, axis=1)
            hm_y_vec = tf.reduce_sum(hm_softmax, axis=2)

            hm_x = tf.reduce_sum(self.heat_vec * hm_x_vec, axis=1, keepdims=True)
            hm_y = tf.reduce_sum(self.heat_vec * hm_y_vec, axis=1, keepdims=True)

            return tf.transpose(tf.concat([hm_x, hm_y], axis=1), perm=[0, 2, 1])

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

            for l_idx in range(self.nStacks):
                with tf.variable_scope("hg_level_{}".format(l_idx)):
                    hg = hourglass.build_hourglass(inter, self.nFeats, 4, name="hg", is_training=self.is_training, nModules=self.nModules, res_utils=self.res_utils)
                    ll = hg
                    ll = mConvBnRelu(inputs=ll, nOut=self.nFeats, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin")

                    tmp_hm = tf.layers.conv2d(inputs=ll, filters=self.nJoints, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="heatmaps")
                    tmp_int = self.build_integral(input_heatmap=tmp_hm, name="integrals")

                    self.heatmaps.append(tmp_hm)
                    self.integrals.append(tmp_int)
                    ## Add predictions back if this is not the last hg module
                    if len(self.heatmaps) < self.nStacks:
                        with tf.variable_scope("add_block"):
                            ll_ = tf.layers.conv2d(inputs=ll, filters=self.nFeats, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="ll_")
                            tmp_hm_ = tf.layers.conv2d(inputs=tmp_hm, filters=self.nFeats, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="tmp_hm_")
                            inter = tf.add_n([inter, ll_, tmp_hm_])

    # input_joints shape (None, 17, 2)
    def build_input_heatmaps(self, input_center, stddev=1.0, name="input_heatmaps", gaussian_coefficient=True):
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

    def build_loss(self, input_heatmap, input_joints_2d, lr, lr_decay_step, lr_decay_rate, use_l2=True):

        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        self.total_loss = 0
        with tf.variable_scope("losses"):
            with tf.variable_scope("heatmap_loss"):
                self.heatmap_losses = []
                for idx in range(self.nStacks):
                    hm_loss = tf.nn.l2_loss(input_heatmap - self.heatmaps[idx], name="heatmap_loss_{}".format(idx)) / self.batch_size * self.loss_weight_heatmap
                    self.heatmap_losses.append(hm_loss)
                    # self.total_loss = self.total_loss + hm_loss

            with tf.variable_scope("integral_loss"):
                self.integral_losses = []
                for idx in range(self.nStacks):
                    # self.integral_loss.append(m_l1_loss(cur_integral_2d - input_joints_2d, name="integral_loss_{}".format(idx)) * self.loss_weight_integral)
                    if use_l2:
                        int_loss = tf.nn.l2_loss(input_joints_2d - self.integrals[idx], name="integral_loss_{}".format(idx)) / self.batch_size * self.loss_weight_integral
                    else:
                        int_loss = m_l1_loss(input_joints_2d - self.integrals[idx], name="integral_loss_{}".format(idx)) / self.batch_size * self.loss_weight_integral

                    self.integral_losses.append(int_loss)
                    self.total_loss = self.total_loss + int_loss

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
                combined_heatmaps = tf.concat([input_heatmap, self.heatmaps[-1]], axis=0)
                hm_all_joints_2d = self.get_joints_hm(combined_heatmaps, batch_size=2*cur_batch_size, name="heatmap_to_joints")

                hm_gt_2d = hm_all_joints_2d[0:cur_batch_size] * self.pose_2d_scale
                hm_pd_2d = hm_all_joints_2d[cur_batch_size:] * self.pose_2d_scale

                self.heatmaps_acc = self.cal_accuracy(gt_joints=hm_gt_2d, pd_joints=hm_pd_2d, name="joints_2d_acc")

                int_gt_2d = input_joints_2d * self.pose_2d_scale
                int_pd_2d = self.integrals[-1] * self.pose_2d_scale

                self.integral_acc = self.cal_accuracy(gt_joints=int_gt_2d, pd_joints=int_pd_2d, name="integral_2d_acc")

        tf.summary.scalar("total_loss_scalar", self.total_loss)
        for i in range(self.nStacks):
            tf.summary.scalar("heatmaps_loss_{}".format(i), self.heatmap_losses[i])

        for i in range(self.nStacks):
            tf.summary.scalar("integral_loss_{}".format(i), self.integral_losses[i])

        tf.summary.scalar("heatmaps_acc_scalar", self.heatmaps_acc)
        tf.summary.scalar("integral_acc_scalar", self.integral_acc)
        tf.summary.scalar("learning_rate", self.lr)
        self.merged_summary = tf.summary.merge_all()
