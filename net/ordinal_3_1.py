import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
from network_utils import mConvBnRelu
import hourglass

# The structure is translate from github.com/umich-vl/pose-hg-train/blob/maskter/src/models/hg.lua

# is_training is a tensor or python bool
class mOrdinal_3_1(object):
    def __init__(self, nJoints, is_training, batch_size, img_size=256, depth_scale=1000.0, loss_weight=1000):
        self.nJoints = nJoints
        self.img_size = img_size
        self.is_use_bias = True
        self.is_tiny = False
        self.is_training = is_training
        self.res_utils = mResidualUtils(is_training=self.is_training, is_use_bias=self.is_use_bias, is_tiny=self.is_tiny)
        self.batch_size = batch_size
        self.depth_scale = depth_scale
        self.loss_weight = loss_weight

    def build_model(self, input_images):
        with tf.variable_scope("ordinal_3_1"):

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
                                 kernel_initializer=tf.initializers.truncated_normal(stddev=0.01),
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
                self.result = tf.layers.dense(inputs=net, units=self.nJoints, activation=None, kernel_initializer=tf.initializers.truncated_normal(stddev=0.0001), name="fc")

    def cal_accuracy(self, gt_depth, pd_depth):
        accuracy = tf.reduce_mean(tf.abs(self.depth_scale * gt_depth - self.depth_scale * pd_depth))
        return accuracy

    # ordinal_3_1 with no ground truth
    def build_loss_gt(self, input_depth, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        self.loss = tf.nn.l2_loss(input_depth - self.result, name="depth_l2_loss") / self.batch_size * self.loss_weight

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
            self.accuracy = self.cal_accuracy(input_depth, self.result)

        tf.summary.scalar("depth_accuracy(mm)", self.accuracy)
        tf.summary.scalar("depth_l2_loss", self.loss)
        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()

    def build_loss_no_gt(self, gt_depth, relation_table, loss_table_log, loss_table_pow, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        with tf.device("/device:GPU:0"):
            self.loss = 0
            with tf.variable_scope("rank_loss"):
                row_val = tf.tile(self.result[:, :, tf.newaxis], [1, 1, self.nJoints])
                col_val = tf.tile(self.result[:, tf.newaxis], [1, self.nJoints, 1])

                rel_distance = (row_val - col_val)
                self.rank_loss = tf.reduce_sum(loss_table_log * tf.log(1 + tf.exp(relation_table * rel_distance)) + loss_table_pow * tf.pow(rel_distance, 2)) / self.batch_size

            ############## Test ordinal supervision with gt supervision ##############
            with tf.variable_scope("gt_loss"):
                self.gt_loss = tf.nn.l2_loss(gt_depth - self.result, name="l2_loss") / self.batch_size

            self.loss = self.rank_loss + self.gt_loss
            ##########################################################################

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
            self.accuracy = self.cal_accuracy(input_depth, self.result)

        tf.summary.scalar("depth_accuracy(mm)", self.accuracy)
        tf.summary.scalar("rank_loss", self.loss)
        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()
