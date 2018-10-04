import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
from network_utils import mConvBnRelu
import hourglass

# The structure is translate from github.com/umich-vl/pose-hg-train/blob/maskter/src/models/hg.lua

# is_training is a tensor or a python bool
class mOrdinal_3_2(object):

    def __init__(self, nJoints, is_training, batch_size, img_size=256, coords_scale=1000.0, coords_scale_2d=255.0, keyp_loss_weight=100.0, rank_loss_weight=1.0, coords_loss_weight=1000.0):
        self.nJoints = nJoints
        self.img_size = img_size
        self.is_use_bias = True
        self.is_tiny = False
        self.is_training = is_training
        self.res_utils = mResidualUtils(is_training=self.is_training, is_use_bias=self.is_use_bias, is_tiny=self.is_tiny)
        self.batch_size = batch_size
        self.coords_scale = coords_scale
        self.coords_scale_2d = coords_scale_2d

        self.coords_loss_weight = coords_loss_weight
        self.keyp_loss_weight = keyp_loss_weight
        self.rank_loss_weight = rank_loss_weight

    def build_model(self, input_images):
        with tf.variable_scope("ordinal_3_2"):

            with tf.variable_scope("conv1"):
                first_conv = tf.layers.conv2d(
                                 inputs=input_images,
                                 filters=64,
                                 kernel_size=7,
                                 strides=2,
                                 padding="SAME",
                                 use_bias=self.is_use_bias,
                                 activation=None,
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name="conv")
                first_conv = tf.contrib.layers.batch_norm(first_conv, 0.9, center=True, scale=True, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.is_training)

            net = self.res_utils.residual_block(first_conv, 128, name="res1")
            net = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net, 128, name="res2")
            net = self.res_utils.residual_block(net, 128, name="res3")
            net = self.res_utils.residual_block(net, 256, name="res4")

            net = hourglass.build_hourglass(net, 512, 4, name="hg_1", is_training=self.is_training, res_utils=self.res_utils)
            net = mConvBnRelu(inputs=net, nOut=512, kernel_size=1, strides=1, name="lin1", is_training=self.is_training, is_use_bias=self.is_use_bias)
            net = mConvBnRelu(inputs=net, nOut=256, kernel_size=1, strides=1, name="lin2", is_training=self.is_training, is_use_bias=self.is_use_bias)

            with tf.variable_scope("final_fc"):
                net = tf.layers.flatten(net)
                self.result = tf.layers.dense(inputs=net, units=self.nJoints*3, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc")
                self.result = tf.reshape(self.result, [-1, self.nJoints, 3])

    def cal_accuracy(self, gt_joints, pd_joints):
        accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(self.coords_scale * gt_joints - self.coords_scale * pd_joints, 2), axis=2)))
        return accuracy

    def cal_accuracy_2d(self, gt_joints_2d, pd_joints_2d):
        accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(self.coords_scale_2d * gt_joints_2d - self.coords_scale_2d * pd_joints_2d, 2), axis=2)))
        return accuracy

    # The fully supervision of the 3d coords: joints are all related to root!
    def build_loss_gt(self, input_coords, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        self.loss = tf.nn.l2_loss(input_coords - self.result, name="coords_l2_loss") / self.batch_size * self.coords_loss_weight

        # NOTICE: The dependencies must be added, because of the BN used in the residual 
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm

        # for g, v in grads_n_vars:
            # tf.summary.histogram(v.name, v)
            # tf.summary.histogram(v.name+"_grads", g)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Update_ops num {}".format(len(update_ops)))
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, self.global_steps)

        with tf.variable_scope("cal_accuracy"):
            self.accuracy = self.cal_accuracy(input_coords, self.result)

        tf.summary.scalar("coords_accuracy(mm)", self.accuracy)
        tf.summary.scalar("coords_l2_loss", self.loss)
        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()

    def build_loss_no_gt(self, input_coords_2d, relation_table, loss_table_log, loss_table_pow, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        with tf.device("/device:GPU:0"):
            with tf.variable_scope("loss_calculation"):
                self.loss = 0

                result_coords_2d = self.result[:, :, 0:2]
                result_depth = self.result[:, :, 2]

                with tf.variable_scope("rank_loss"):
                    row_val = tf.tile(self.result[:, :, tf.newaxis], [1, 1, self.nJoints])
                    col_val = tf.tile(self.result[:, tf.newaxis], [1, self.nJoints, 1])

                    rel_distance = (row_val - col_val)
                    # Softplus is log(1 + exp(x)) and without overflow
                    self.rank_loss = tf.reduce_sum(loss_table_log * tf.math.softplus(relation_table * rel_distance) + loss_table_pow * tf.pow(rel_distance, 2)) / self.batch_size * self.rank_loss_weight

                with tf.variable_scope("coord2d_loss"):
                    self.coord2d_loss = tf.nn.l2_loss(result_coords_2d - input_coords_2d) / self.batch_size * self.keyp_loss_weight

                self.loss = self.coord2d_loss + self.rank_loss

            with tf.variable_scope("optimizer"):
                # NOTICE: The dependencies must be added, because of the BN used in the residual 
                # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                print("Update_ops num {}".format(len(update_ops)))
                with tf.control_dependencies(update_ops):
                    self.train_op = self.optimizer.minimize(self.loss, self.global_steps)

        with tf.variable_scope("cal_accuracy_2d"):
            self.accuracy_2d = self.cal_accuracy_2d(input_coords_2d, result_coords_2d)

        with tf.device("/cpu:0"):
            tf.summary.scalar("coords_2d_accuracy(px)", self.accuracy_2d)
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("rank_loss_curve", self.rank_loss)
            tf.summary.scalar("coord2d_loss_curve", self.coord2d_loss)
            tf.summary.scalar("learning_rate", self.lr)

            self.merged_summary = tf.summary.merge_all()
