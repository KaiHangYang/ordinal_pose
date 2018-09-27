import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
import hourglass

# The structure is translate from github.com/umich-vl/pose-hg-train/blob/maskter/src/models/hg.lua

class mOrdinal_3_1(object):
    def __init__(self, nJoints, img_size=256, batch_size=4, is_training=True):
        self.nJoints = nJoints
        self.img_size = img_size
        self.is_use_bias = True
        self.is_tiny = False
        self.is_training = is_training
        self.res_utils = mResidualUtils(is_training=self.is_training, is_use_bias=self.is_use_bias, is_tiny=self.is_tiny)
        self.batch_size = batch_size

    def build_model(self, input_images):
        with tf.variable_scope("ordinal_3_1"):

            with tf.variable_scope("res1"):
                first_conv = tf.layers.conv2d(
                                 inputs=input_images,
                                 filters=64,
                                 kernel_size=7,
                                 strides=2,
                                 padding="SAME",
                                 use_bias=self.is_use_bias,
                                 activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name="conv")
                first_conv = tf.contrib.layers.batch_norm(first_conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.is_training)

            net = self.res_utils.residual_block(first_conv, 128, name="res2")
            net = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net, 128, name="res3")
            net = self.res_utils.residual_block(net, 256, name="res4")

            net = hourglass.build_hourglass(net, 256, 4, name="hg_1", is_training=self.is_training, res_utils=self.res_utils)
            net = self.res_utils.residual_block(net, 256, name="out_res")

            with tf.variable_scope("final_fc"):
                features_shape = net.get_shape().as_list()
                net = tf.reshape(net, [features_shape[0], -1])
                net = tf.layers.dropout(net, rate=0.2, name="dropout")
                self.result = tf.layers.dense(inputs=net, units=self.nJoints, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc")

    def cal_accuracy(self, gt_depth, pd_depth):
        accuracy = tf.reduce_mean(tf.abs(gt_depth - pd_depth))
        return accuracy

    # ordinal_3_1 with no ground truth
    def build_loss_gt(self, input_depth, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        self.loss = tf.nn.l2_loss(input_depth - self.result, name="depth_l2_loss") / self.batch_size

        # NOTICE: The dependencies must be added, because of the BN used in the residual 
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        grads_n_vars = self.optimizer.compute_gradients(self.loss)
        # for g, v in grads_n_vars:
            # tf.summary.histogram(v.name, v)
            # tf.summary.histogram(v.name+"_grads", g)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.apply_gradients(grads_n_vars, self.global_steps)

        with tf.variable_scope("cal_accuracy"):
            self.accuracy = self.cal_accuracy(input_depth, self.result)

        tf.summary.scalar("depth_accuracy(mm)", self.accuracy)
        tf.summary.scalar("depth_l2_loss", self.loss)
        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()

    # def build_loss_no_gt(self, relation_table, loss_table_log, loss_table_pow, lr):
        # self.global_steps = tf.train.get_or_create_global_step()

        # with tf.device("/device:GPU:0"):
            # with tf.variable_scope("rank_loss"):
                # self.loss = 0
                # row_val = tf.tile(self.result[:, :, tf.newaxis], [1, 1, self.nJoints])
                # col_val = tf.tile(self.result[:, tf.newaxis], [1, self.nJoints, 1])

                # rel_distance = (row_val - col_val)

                # self.loss = tf.reduce_sum(loss_table_log * tf.log(1 + tf.exp(relation_table * rel_distance)) + loss_table_pow * tf.pow(rel_distance, 2)) / self.batch_size

            # with tf.variable_scope("grad"):

                # # NOTICE: The dependencies must be added, because of the BN used in the residual 
                # # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
                # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):
                    # grads_n_vars = self.optimizer.compute_gradients(self.loss)
                    # self.train_op = self.optimizer.apply_gradients(grads_n_vars, self.global_steps)

        # with tf.device("/cpu:0"):
            # tf.summary.scalar("loss", self.loss)
            # self.merged_summary = tf.summary.merge_all()
