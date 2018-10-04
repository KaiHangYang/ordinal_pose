import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
from network_utils import mConvBnRelu
import hourglass

# The structure is translate from github.com/umich-vl/pose-hg-train/blob/maskter/src/models/hg.lua

# is_training is a tensor or python bool
class mOrdinal_3_3(object):
    def __init__(self, nJoints, is_training, batch_size, img_size=256, loss_weight_volume=1.0):
        self.loss_weight_volume = loss_weight_volume
        self.nJoints = nJoints
        self.img_size = img_size
        self.is_use_bias = True
        self.is_tiny = False
        self.is_training = is_training
        self.res_utils = mResidualUtils(is_training=self.is_training, is_use_bias=self.is_use_bias, is_tiny=self.is_tiny)
        self.batch_size = batch_size


    # copy the implementation from https://github.com/geopavlakos/c2f-vol-train/blob/master/src/models/hg-stacked.lua
    def build_model(self, input_images):
        with tf.variable_scope("ordinal_3_3"):
            net = mConvBnRelu(inputs=input_images, nOut=64, kernel_size=7, strides=2, is_use_bias=self.is_use_bias, is_training=self.is_training, name="conv1")

            net = self.res_utils.residual_block(net, 128, name="res1")
            net_pooled = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net_pooled, 128, name="res2")
            net = self.res_utils.residual_block(net, 128, name="res3")
            net = self.res_utils.residual_block(net, 256, name="res4")

            hg1 = hourglass.build_hourglass(net, 512, 4, name="hg_1", is_training=self.is_training, res_utils=self.res_utils)

            lin1 = mConvBnRelu(inputs=hg1, nOut=512, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="lin1")
            lin2 = mConvBnRelu(inputs=lin1, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="lin2")

            self.volumes = tf.layers.conv2d(inputs=lin2, filters=self.nJoints*64, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="volumes")


    def cal_accuracy(self, gt_volume, pd_volume):
        # accuracy = tf.reduce_mean(tf.abs(self.depth_scale * gt_depth - self.depth_scale * pd_depth))
        # return accuracy
        pass

    # ordinal_3_3 with ground true volumes
    def build_loss_gt(self, input_volumes, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        with tf.variable_scope("loss"):
            self.loss = tf.nn.l2_loss(input_volumes - self.volumes, name="volume_loss") / self.batch_size * self.loss_weight_volume

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

        # with tf.variable_scope("cal_accuracy"):
            # self.accuracy = self.cal_accuracy(input_depth, self.result)

        # tf.summary.scalar("depth_accuracy(mm)", self.accuracy)
        tf.summary.scalar("loss", self.loss)
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
