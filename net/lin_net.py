import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
from network_utils import mConvBnRelu
import hourglass

# The structure is translate from github.com/umich-vl/pose-hg-train/blob/maskter/src/models/hg.lua
# is_training is a tensor or python bool
class mLinNet(object):
    def __init__(self, nJoints, is_training, batch_size, pose_3d_scale=1000):
        self.model_name = "LinNet"

        self.nJoints = nJoints
        self.pose_3d_scale = pose_3d_scale
        # self.is_use_bias = True
        # self.is_tiny = False
        # self.is_use_bn = is_use_bn
        self.is_training = is_training
        self.batch_size = batch_size
        self.keep_prob = 0.5
        self.poses = [None, None]

    def build_model(self, input_arr):
        with tf.variable_scope(self.model_name):

            initial_features = tf.layers.dense(inputs=input_arr, units=1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="initial_fc")
            initial_features = tf.layers.batch_normalization(initial_features, training=self.is_training)
            initial_features = tf.nn.dropout(initial_features, keep_prob=self.keep_prob)
            initial_features = tf.nn.relu(initial_features)

            with tf.variable_scope("block_1"):
                net = tf.layers.dense(inputs=initial_features, units=1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                net = tf.nn.relu(net)

                net = tf.layers.dense(inputs=net, units=1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc2")
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                net = tf.nn.relu(net)

                block_features = net+initial_features

                net = tf.layers.dense(inputs=block_features, units=1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc3")
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                net = tf.nn.relu(net)

                net = tf.layers.dense(inputs=net, units=1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc4")
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                net = tf.nn.relu(net)

                inter_features = net + block_features

            with tf.variable_scope("output_1"):
                cur_pose = tf.layers.dense(inputs=inter_features, units=3*self.nJoints, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out1")
                self.poses[0] = tf.reshape(cur_pose, [-1, self.nJoints, 3])

                net = tf.layers.dense(inputs=cur_pose, units=1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out1_features")
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                net = tf.nn.relu(net)

                inter_features = inter_features + net + initial_features

            with tf.variable_scope("block_2"):
                net = tf.layers.dense(inputs=inter_features, units=1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                net = tf.nn.relu(net)

                net = tf.layers.dense(inputs=net, units=1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc2")
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                net = tf.nn.relu(net)

                block_features = net + inter_features

                net = tf.layers.dense(inputs=block_features, units=1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc3")
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                net = tf.nn.relu(net)

                net = tf.layers.dense(inputs=net, units=1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc4")
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                net = tf.nn.relu(net)

                final_features = initial_features + block_features + net

            with tf.variable_scope("output_2"):
                self.poses[1] = tf.reshape(tf.layers.dense(inputs=final_features, units=3*self.nJoints, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out2"), [-1, self.nJoints, 3])


    def cal_accuracy(self, gt_joints, pd_joints, name="accuracy"):
        with tf.variable_scope(name):
            accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(gt_joints - pd_joints, 2), axis=2)))
        return accuracy

    def build_evaluation(self):
        self.global_steps = tf.train.get_or_create_global_step()
        with tf.variable_scope("parser_pose"):
            self.pd_poses_0 = (self.poses[0] - tf.tile(self.poses[0][:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_3d_scale
            self.pd_poses = (self.poses[-1] - tf.tile(self.poses[-1][:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_3d_scale

    def build_loss(self, input_poses, lr, lr_decay_step=None, lr_decay_rate=None):
        self.global_steps = tf.train.get_or_create_global_step()

        if lr_decay_step is None or lr_decay_rate is None:
            self.lr = lr
        else:
            self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase=True, name= 'learning_rate')

        self.pose_loss = []
        self.total_loss = 0

        with tf.variable_scope("losses"):
            for pose_level in range(len(self.poses)):
                self.pose_loss.append(tf.nn.l2_loss(input_poses - self.poses[pose_level], name="pose_loss_{}".format(pose_level)) / self.batch_size)
                self.total_loss += self.pose_loss[-1]

        # NOTICE: The dependencies must be added, because of the BN used in the residual 
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Update_ops num {}".format(len(update_ops)))
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.total_loss, self.global_steps)

        with tf.variable_scope("parser_results"):
            cur_batch_size = tf.cast(self.batch_size, dtype=tf.int32)
            with tf.variable_scope("parser_pose"):
                self.gt_poses = (input_poses - tf.tile(input_poses[:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_3d_scale
                self.pd_poses = (self.poses[-1] - tf.tile(self.poses[-1][:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_3d_scale
                self.pd_poses_0 = (self.poses[0] - tf.tile(self.poses[0][:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_3d_scale

        self.accuracy_pose = []
        with tf.variable_scope("cal_accuracy"):
            self.accuracy_pose.append(self.cal_accuracy(self.gt_poses, self.pd_poses_0, name="poses_accuracy_0"))
            self.accuracy_pose.append(self.cal_accuracy(self.gt_poses, self.pd_poses, name="poses_accuracy"))

        tf.summary.scalar("pose_accuracy", self.accuracy_pose[-1])
        tf.summary.scalar("total_loss_scalar", self.total_loss)

        for idx, cur_pose_loss in enumerate(self.pose_loss):
            tf.summary.scalar("pose_loss_{}".format(idx), cur_pose_loss)

        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()
