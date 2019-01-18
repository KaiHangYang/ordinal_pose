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

        self.poses = [None, None]

    # copy the implementation from https://github.com/geopavlakos/c2f-vol-train/blob/master/src/models/hg-stacked.lua
    def build_model(self, input_arr):
        with tf.variable_scope(self.model_name):

            initial_features = tf.layers.dense(inputs=input_arr, units=1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="initial_fc")

            with tf.variable_scope("block_1"):
                net = tf.layers.dense(inputs=initial_features, units=1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")
                net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc2")

                block_features = net+initial_features

                net = tf.layers.dense(inputs=block_features, units=1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc3")
                net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc4")

                inter_features = net + block_features

            with tf.variable_scope("output_1"):
                cur_pose = tf.layers.dense(inputs=inter_features, units=3*self.nJoints, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out1")
                self.poses[0] = tf.reshape(cur_pose, [-1, self.nJoints, 3])
                net = tf.layers.dense(inputs=cur_pose, units=1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out1_features")

                inter_features = inter_features + net + initial_features

            with tf.variable_scope("block_2"):
                net = tf.layers.dense(inputs=inter_features, units=1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")
                net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc2")

                block_features = net + inter_features

                net = tf.layers.dense(inputs=block_features, units=1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc3")
                net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc4")

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

        with tf.variable_scope("cal_accuracy"):
            self.accuracy_pose = self.cal_accuracy(self.gt_poses, self.pd_poses, name="poses_accuracy")

        tf.summary.scalar("pose_accuracy", self.accuracy_pose)
        tf.summary.scalar("total_loss_scalar", self.total_loss)

        for idx, cur_pose_loss in enumerate(self.pose_loss):
            tf.summary.scalar("pose_loss_{}".format(idx), cur_pose_loss)

        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()
