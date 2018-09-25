import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
import hourglass


# The structure is translate from github.com/umich-vl/pose-hg-train/blob/maskter/src/models/hg.lua

class mOrdinal_3_2(object):
    def __init__(self, nJoints, img_size=256, batch_size=4, is_training=True):
        self.nJoints = nJoints
        self.img_size = img_size
        self.is_use_bias = True
        self.is_tiny = False

        self.res_utils = mResidualUtils(is_training=is_training, is_use_bias=self.is_use_bias, is_tiny=self.is_tiny)
        self.is_training = is_training
        self.batch_size = batch_size

    def build_model(self, input_images):
        with tf.variable_scope("ordinal_3_2"):
            with tf.variable_scope("res1"):
                first_conv = tf.layers.conv2d(
                                 inputs=input_images,
                                 filters=64,
                                 kernel_size=7,
                                 strides=2,
                                 padding="SAME",
                                 activation=None,
                                 use_bias=self.is_use_bias
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name="conv")
                first_conv = tf.contrib.layers.batch_norm(first_conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.is_training, updates_collections=None)

            net = self.res_utils.residual_block(first_conv, 128, name="res2")
            net = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net, 128, name="res3")
            net = self.res_utils.residual_block(net, 256, name="res4")

            net = hourglass.build_hourglass(net, 256, 4, name="hg_1", is_training=self.is_training, res_utils=self.res_utils)
            net = self.res_utils.residual_block(net, 256, name="out_res")

            with tf.variable_scope("final_fc"):
                features_shape = net.get_shape().as_list()
                net = tf.reshape(net, [features_shape[0], -1])
                self.result = tf.contrib.layers.fully_connected(inputs=net, num_outputs=3*self.nJoints, activation_fn=None)
                self.result = tf.reshape(self.result, [features_shape[0], self.nJoints, 3])

    def build_loss_gt(self, input_coords, lr):
        self.global_steps = tf.train.get_or_create_global_step()
        with tf.variable_scope("loss_calculation"):
            self.loss = tf.nn.l2_loss(input_coords - self.result, name="ordinal_3_2_gt_loss") / self.batch_size

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        grads_n_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(grads_n_vars, self.global_steps)
        tf.summary.scalar("gt_coords_loss", self.loss)

        self.merged_summary = tf.summary.merge_all()


    def build_loss_no_gt(self, input_coords_2d, relation_table, loss_table_log, loss_table_pow, lr):
        self.global_steps = tf.train.get_or_create_global_step()

        with tf.device("/device:GPU:0"):
            with tf.variable_scope("loss_calculation"):
                self.loss = 0
                result_coords_2d = self.result[:, :, 0:2]
                result_depth = self.result[:, :, 2]

                with tf.variable_scope("rank_loss"):
                    row_val = tf.tile(result_depth[:, :, tf.newaxis], [1, 1, self.nJoints])
                    col_val = tf.tile(result_depth[:, tf.newaxis], [1, self.nJoints, 1])

                    rel_distance = (row_val - col_val)
                    self.rank_loss = tf.reduce_sum(loss_table_log * tf.log(1 + tf.exp(relation_table * rel_distance)) + loss_table_pow * tf.pow(rel_distance, 2)) / self.batch_size

                with tf.variable_scope("coord2d_loss"):
                    self.coord2d_loss = tf.nn.l2_loss(result_coords_2d - input_coords_2d)

                self.loss = 100 * self.coord2d_loss + self.rank_loss

            with tf.variable_scope("grad"):
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
                grads_n_vars = self.optimizer.compute_gradients(self.loss)
                self.train_op = self.optimizer.apply_gradients(grads_n_vars, self.global_steps)

        with tf.device("/cpu:0"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("rank_loss_curve", self.rank_loss)
            tf.summary.scalar("coord2d_loss_curve", 100 * self.coord2d_loss)
            self.merged_summary = tf.summary.merge_all()
