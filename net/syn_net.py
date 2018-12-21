import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
from network_utils import mConvBnRelu
from network_utils import m_l1_loss
import hourglass

# copy the implementation from https://github.com/geopavlakos/c2f-vol-train/blob/master/src/models/hg-stacked.lua

# is_training is a tensor or python bool
class mSynNet(object):
    def __init__(self, nJoints, is_training, batch_size, img_size=256, loss_weight_heatmap=1.0, loss_weight_fb=1.0, loss_weight_br=1.0, pose_2d_scale=4, nModules=2, nStacks=2, nRegModules=2, nFeats=256, is_use_bn=True, zero_debias_moving_mean=False):
        self.model_name = "SynNet"

        self.loss_weight_heatmap = loss_weight_heatmap
        self.loss_weight_fb = loss_weight_fb
        self.loss_weight_br = loss_weight_br
        self.pose_2d_scale = pose_2d_scale

        self.nJoints = nJoints
        self.img_size = img_size
        self.is_use_bias = True
        self.is_tiny = False
        self.is_use_bn = is_use_bn
        self.is_training = is_training
        self.zero_debias_moving_mean = zero_debias_moving_mean
        self.res_utils = mResidualUtils(is_training=self.is_training, is_use_bias=self.is_use_bias, is_tiny=self.is_tiny, is_use_bn=self.is_use_bn, zero_debias_moving_mean=self.zero_debias_moving_mean)
        self.batch_size = batch_size
        self.feature_size = 64
        self.nModules = nModules
        self.nStacks = nStacks
        self.nRegModules = nRegModules
        self.nFeats = nFeats
        self.heatmaps = []

    # copy the implementation from https://github.com/geopavlakos/c2f-vol-train/blob/master/src/models/hg-stacked.lua
    def build_model(self, input_images):
        with tf.variable_scope(self.model_name):
            net = mConvBnRelu(inputs=input_images, nOut=64, kernel_size=7, strides=2, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="conv1", zero_debias_moving_mean=self.zero_debias_moving_mean)

            net = self.res_utils.residual_block(net, 128, name="res1")
            net_pooled = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net_pooled, 128, name="res2")
            net = self.res_utils.residual_block(net, self.nFeats, name="res3")
            inter = net

            for s_idx in range(self.nStacks):
                with tf.variable_scope("hg_block_{}".format(s_idx)):
                    hg = hourglass.build_hourglass(inter, self.nFeats, 4, name="hg", is_training=self.is_training, nModules=self.nModules, res_utils=self.res_utils)
                    ll = hg
                    with tf.variable_scope("ll_block"):
                        for i in range(self.nModules):
                            ll = self.res_utils.residual_block(ll, self.nFeats, name="res{}".format(i))
                        ll = mConvBnRelu(inputs=ll, nOut=self.nFeats, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin", zero_debias_moving_mean=self.zero_debias_moving_mean)

                    with tf.variable_scope("out"):
                        # output the heatmaps
                        self.heatmaps.append(tf.layers.conv2d(inputs=ll, filters=self.nJoints, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="heatmaps"))

                        # add the features
                        out_ = tf.layers.conv2d(inputs=self.heatmaps[s_idx], filters=self.nFeats, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out_")
                        ll_ = tf.layers.conv2d(inputs=ll, filters=self.nFeats, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="ll_")

                        inter = tf.add_n([inter, ll_, out_])
            reg = inter

            for i in range(4):
                with tf.variable_scope("final_downsample_{}".format(i)):
                    for j in range(self.nRegModules):
                        reg = self.res_utils.residual_block(reg, self.nFeats, name="res{}".format(j))
                    reg = tf.layers.max_pooling2d(reg, pool_size=2, strides=2, padding="VALID", name="maxpool")

            with tf.variable_scope("final_classify"):
                reg = tf.layers.flatten(reg)
                # fb information and br matrix (half matrix)
                self.results = tf.layers.dense(inputs=reg, units= 3 * self.nJoints * (self.nJoints - 1) / 2, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc")
                self.results = tf.reshape(self.results, [self.batch_size, -1, 3])

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

    def build_evaluation(self):

        # 1 is forward, 0 is uncertain, -1 is backward
        self.fb_info = self.results[:, 0:self.nJoints-1]
        # 1 is in front, 0 is unconcerned or unknown, -1 is behind
        self.br_info = self.results[:, self.nJoints-1:]

        with tf.variable_scope("extract_heatmap"):
            cur_batch_size = tf.cast(self.batch_size, dtype=tf.int32)
            self.pd_2d = self.get_joints_hm(self.heatmaps[-1], batch_size=cur_batch_size, name="heatmap_to_joints") * self.pose_2d_scale
        with tf.variable_scope("extract_fb"):
            self.pd_fb_result = tf.argmax(self.fb_info, axis=2)
            self.pd_fb_belief = tf.reduce_max(self.fb_info, axis=2)

        with tf.variable_scope("extract_br"):
            self.pd_br_result = tf.argmax(self.br_info, axis=2)
            self.pd_br_belief = tf.reduce_max(self.br_info, axis=2)

    def build_loss(self, input_heatmaps, input_fb, input_br, lr, lr_decay_step=None, lr_decay_rate=None):
        self.global_steps = tf.train.get_or_create_global_step()

        if lr_decay_step is None or lr_decay_rate is None:
            self.lr = lr
        else:
            self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        with tf.variable_scope("losses"):
            # 1 is forward, 0 is uncertain, -1 is backward
            self.fb_info = self.results[:, 0:self.nJoints-1]
            # 1 is in front, 0 is unconcerned or unknown, -1 is behind
            self.br_info = self.results[:, self.nJoints-1:]

            self.fb_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_fb, logits=self.fb_info, dim=-1, name="fb_loss")) / self.batch_size * self.loss_weight_fb
            self.br_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_br, logits=self.br_info, dim=-1, name="br_loss")) / self.batch_size * self.loss_weight_br

            self.heatmaps_loss = []
            self.total_loss = 0
            for i in range(len(self.heatmaps)):
                tmp_loss = tf.nn.l2_loss(input_heatmaps - self.heatmaps[i], name="heatmaps_loss_{}".format(i)) / self.batch_size * self.loss_weight_heatmap
                self.heatmaps_loss.append(tmp_loss)
                self.total_loss += tmp_loss

            self.total_loss = self.fb_loss + self.br_loss + self.total_loss

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
                combined_heatmaps = tf.concat([input_heatmaps, self.heatmaps[-1]], axis=0)

                all_joints_2d = self.get_joints_hm(combined_heatmaps, batch_size=2*cur_batch_size, name="heatmap_to_joints")
                self.gt_2d = all_joints_2d[0:cur_batch_size] * self.pose_2d_scale
                self.pd_2d = all_joints_2d[cur_batch_size:] * self.pose_2d_scale

                self.heatmaps_acc = self.cal_accuracy(gt_joints=self.gt_2d, pd_joints=self.pd_2d, name="joints_2d_acc")
            ######### classify accuracy
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

        tf.summary.scalar("total_loss_scalar", self.total_loss)
        for idx in range(len(self.heatmaps)):
            tf.summary.scalar("heatmaps_loss_scalar_level_{}".format(idx), self.heatmaps_loss[idx])

        tf.summary.scalar("fb_loss_scalar", self.fb_loss)
        tf.summary.scalar("br_loss_scalar", self.br_loss)
        tf.summary.scalar("heatmaps_acc_scalar", self.heatmaps_acc)
        tf.summary.scalar("fb_acc_scalar", self.fb_acc)
        tf.summary.scalar("br_acc_scalar", self.br_acc)

        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()
