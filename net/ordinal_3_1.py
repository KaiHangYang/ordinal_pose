import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
import hourglass

class mOrdinal_3_1(object):
    def __init__(self, nJoints, img_size=256, batch_size=4, is_training=True):
        self.nJoints = nJoints
        self.img_size = img_size
        self.res_utils = mResidualUtils(is_training=is_training)
        self.is_training = is_training
        self.batch_size = batch_size

    def build_model(self, input_images):
        with tf.variable_scope("ordinal_3_1"):
            first_conv = tf.layers.conv2d(
                             inputs=input_images,
                             filters=64,
                             kernel_size=7,
                             strides=2,
                             padding="SAME",
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name="conv1")

            net = self.res_utils.residual_block(first_conv, 128, name="res2")
            net = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net, 128, name="res3")
            net = self.res_utils.residual_block(net, 128, name="res4")

            net = hourglass.build_hourglass(net, 256, 4, name="hg_1", is_training=self.is_training)

            features_shape = net.get_shape().as_list()
            net = tf.reshape(net, [features_shape[0], -1])
            self.result = tf.contrib.layers.fully_connected(inputs = net, num_outputs = self.nJoints, activation_fn = None)

    # ordinal_3_1 with no ground truth
    def build_loss_gt(self, input_depth, lr):
        self.global_steps = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        self.loss = tf.nn.l2_loss(input_depth - self.result, name="depth_l2_loss") / self.batch_size

        grads_n_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(grads_n_vars, self.global_steps)

        tf.summary.scalar("depth_l2_loss", self.loss)

        self.merged_summary = tf.summary.merge_all()

    # +1, if joint i is closer than j
    @staticmethod
    def loss_rank_0(zi, zj):
        return tf.log(1 + tf.exp(zi - zj))
    # -1, if joint i is farther than j
    @staticmethod
    def loss_rank_1(zi, zj):
        return tf.log(1 + tf.exp(-zi + zj))
    # 0, if joint i and j are nearly the same depth
    @staticmethod
    def loss_rank_2(zi, zj):
        return tf.pow(zi - zj, 2)

    def build_loss_no_gt(self, relation_table, lr):
        self.global_steps = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        self.loss = 0

        with tf.variable_scope("rank_loss"):
            for n in range(self.batch_size):
                for i in range(self.nJoints):
                    for j in range(i+1, self.nJoints):
                        cur_loss = tf.cond(tf.equal(relation_table[n, i, j], 1), true_fn=lambda:self.loss_rank_0(self.result[n, i], self.result[n, j]), false_fn=lambda:np.float32(0))
                        cur_loss += tf.cond(tf.equal(relation_table[n, i, j], -1), true_fn=lambda:self.loss_rank_1(self.result[n, i], self.result[n, j]), false_fn=lambda:np.float32(0))
                        cur_loss += tf.cond(tf.equal(relation_table[n, i, j], 0), true_fn=lambda:self.loss_rank_2(self.result[n, i], self.result[n, j]), false_fn=lambda:np.float32(0))
                        self.loss += cur_loss

        grads_n_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(grads_n_vars, self.global_steps)

        tf.summary.scalar("rank_loss", self.loss)
        self.merged_summary = tf.summary.merge_all()
