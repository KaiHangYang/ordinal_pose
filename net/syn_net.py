import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
from network_utils import mConvBnRelu
import hourglass

# The structure is translate from github.com/umich-vl/pose-hg-train/blob/maskter/src/models/hg.lua

# is_training is a tensor or python bool
class mSynNet(object):
    def __init__(self, nJoints, is_training, batch_size, img_size=256, loss_weight_sep_synmaps=1.0, loss_weight_synmap=10.0):

        self.loss_weight_sep_synmaps = loss_weight_sep_synmaps
        self.loss_weight_synmap = loss_weight_synmap

        self.nJoints = nJoints
        self.img_size = img_size
        self.is_use_bias = True
        self.is_tiny = False
        self.is_training = is_training
        self.res_utils = mResidualUtils(is_training=self.is_training, is_use_bias=self.is_use_bias, is_tiny=self.is_tiny)
        self.batch_size = batch_size
        self.feature_size = 64

    # copy the implementation from https://github.com/geopavlakos/c2f-vol-train/blob/master/src/models/hg-stacked.lua
    def build_model(self, input_images):
        with tf.variable_scope("SynNet"):
            net = mConvBnRelu(inputs=input_images, nOut=64, kernel_size=7, strides=2, is_use_bias=self.is_use_bias, is_training=self.is_training, name="conv1")

            net = self.res_utils.residual_block(net, 128, name="res1")
            net_pooled = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net_pooled, 128, name="res2")
            net = self.res_utils.residual_block(net, 128, name="res3")
            net = self.res_utils.residual_block(net, 256, name="res4")

            hg1 = hourglass.build_hourglass(net, 256, 4, name="hg_1", is_training=self.is_training, res_utils=self.res_utils)

            lin1 = mConvBnRelu(inputs=hg1, nOut=512, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="lin1")
            lin2 = mConvBnRelu(inputs=lin1, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="lin2")

            # Output the separated bone maps
            # three channel value !!!!
            self.sep_synmaps = tf.layers.conv2d(inputs=lin2, filters=3*(self.nJoints-1), kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=tf.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="sep_synmaps")
            out1_ = tf.layers.conv2d(inputs=self.sep_synmaps, filters=256+128, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out1_")

            with tf.variable_scope("concat_1"):
                cat1 = tf.concat([lin2, net_pooled], axis=3)
                cat1_ = tf.layers.conv2d(inputs=cat1, filters=256+128, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
                int1 = tf.add_n([cat1_, out1_])

            hg2 = hourglass.build_hourglass(int1, 256, 4, name="hg_2", is_training=self.is_training, res_utils=self.res_utils)

            hg2_out_res1 = self.res_utils.residual_block(hg2, 256, name="hg2_out_res1")
            hg2_out_res2 = self.res_utils.residual_block(hg2_out_res1, 256, name="hg2_out_res2")

            with tf.variable_scope("final_output"):
                cur_shape = hg2_out_res2.get_shape()[1:3].as_list()
                out2_resize1 = tf.image.resize_nearest_neighbor(hg2_out_res2, [cur_shape[0] * 2, cur_shape[1] * 2], name="out2_resize1")
                out2_res1 = self.res_utils.residual_block(out2_resize1, 128, kernel_size=5, name="out2_res1")
                cur_shape = out2_res1.get_shape()[1:3].as_list()
                out2_resize2 = tf.image.resize_nearest_neighbor(out2_res1, [cur_shape[0] * 2, cur_shape[1] * 2], name="out2_resize2")
                out2_res2 = self.res_utils.residual_block(out2_resize2, 64, kernel_size=7, name="out2_res2")

                lin3 = mConvBnRelu(inputs=out2_res2, nOut=128, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="lin3")
                lin4 = mConvBnRelu(inputs=lin3, nOut=128, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="lin4")

                # three channel value !!!!
                self.synmap = tf.layers.conv2d(inputs=lin4, filters=3, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=tf.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="synmap")

    def build_loss(self, input_sep_synmaps, input_synmap, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        with tf.variable_scope("losses"):
            self.sep_synmaps_loss = tf.nn.l2_loss(self.sep_synmaps - input_sep_synmaps, name="sep_synmaps_loss") / self.batch_size * self.loss_weight_sep_synmaps
            self.synmap_loss = tf.nn.l2_loss(self.synmap - input_synmap, name="synmap_loss") / self.batch_size * self.loss_weight_synmap

            self.total_loss = self.sep_synmaps_loss + self.synmap_loss

        # NOTICE: The dependencies must be added, because of the BN used in the residual 
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Update_ops num {}".format(len(update_ops)))
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.total_loss, self.global_steps)

        tf.summary.scalar("total_loss_scalar", self.total_loss)
        tf.summary.scalar("sep_synmaps_loss_scalar", self.sep_synmaps_loss)
        tf.summary.scalar("synmap_loss_scalar", self.synmap_loss)
        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()
