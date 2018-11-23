import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
from network_utils import mConvBnRelu
import hourglass

# The structure is translate from github.com/umich-vl/pose-hg-train/blob/maskter/src/models/hg.lua
# is_training is a tensor or python bool
class mPoseNet(object):
    def __init__(self, nJoints, is_training, batch_size, img_size=256, loss_weight_pose=1.0, pose_scale=1000.0, is_use_bn=True, HG_nFeat=256):

        self.loss_weight_pose = loss_weight_pose
        self.pose_scale = pose_scale

        self.nJoints = nJoints
        self.img_size = img_size
        self.is_use_bias = True
        self.is_tiny = False
        self.is_use_bn = is_use_bn
        self.is_training = is_training
        self.res_utils = mResidualUtils(is_training=self.is_training, is_use_bias=self.is_use_bias, is_tiny=self.is_tiny, is_use_bn=self.is_use_bn)
        self.batch_size = batch_size
        self.feature_size = 64
        self.nModules = 3
        self.nRegModules = 2
        self.HG_nFeat = HG_nFeat

    # copy the implementation from https://github.com/geopavlakos/c2f-vol-train/blob/master/src/models/hg-stacked.lua
    def build_model(self, input_images):
        with tf.variable_scope("PoseNet"):
            net = mConvBnRelu(inputs=input_images, nOut=64, kernel_size=7, strides=2, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="conv1")

            net = self.res_utils.residual_block(net, 128, name="res1")
            net_pooled = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net_pooled, 128, name="res2")
            net = self.res_utils.residual_block(net, 128, name="res3")
            net = self.res_utils.residual_block(net, 256, name="res4")

            hg1 = hourglass.build_hourglass(net, self.HG_nFeat, 4, name="hg_1", is_training=self.is_training, res_utils=self.res_utils)

            lin1 = mConvBnRelu(inputs=hg1, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin1")
            lin2 = mConvBnRelu(inputs=lin1, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin2")

            out1_ = tf.layers.conv2d(inputs=lin2, filters=256+128, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out1_")

            with tf.variable_scope("concat_1"):
                cat1 = tf.concat([lin2, net_pooled], axis=3)
                cat1_ = tf.layers.conv2d(inputs=cat1, filters=256+128, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
                int1 = tf.add_n([cat1_, out1_])

            hg2 = hourglass.build_hourglass(int1, self.HG_nFeat, 4, name="hg_2", is_training=self.is_training, res_utils=self.res_utils)

            lin3 = mConvBnRelu(inputs=hg2, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin3")
            lin4 = mConvBnRelu(inputs=lin3, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin4")

            reg = lin4
            for i in range(4):
                with tf.variable_scope("final_downsample_{}".format(i)):
                    for j in range(self.nRegModules):
                        reg = self.res_utils.residual_block(reg, 256, name="res{}".format(j))
                    reg = tf.layers.max_pooling2d(reg, pool_size=2, strides=2, padding="VALID", name="maxpool")

            with tf.variable_scope("final_pose"):
                reg = tf.layers.flatten(reg)
                # fb information
                self.poses = tf.layers.dense(inputs=reg, units=3*self.nJoints, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc")
                self.poses = tf.reshape(self.poses, [self.batch_size, self.nJoints, 3])

    def cal_accuracy(self, gt_joints, pd_joints, name="accuracy"):
        with tf.variable_scope(name):
            accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(gt_joints - pd_joints, 2), axis=2)))
        return accuracy

    def build_evaluation(self):
        self.global_steps = tf.train.get_or_create_global_step()
        with tf.variable_scope("parser_pose"):
            self.pd_3d = (self.poses - tf.tile(self.poses[:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_scale

    def build_loss(self, input_poses, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        with tf.variable_scope("losses"):
            self.pose_loss = tf.nn.l2_loss(input_poses - self.poses, name="pose_loss") / self.batch_size * self.loss_weight_pose

            self.total_loss = self.pose_loss

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
                self.gt_poses = (input_poses - tf.tile(input_poses[:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_scale
                self.pd_poses = (self.poses - tf.tile(self.poses[:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_scale

        with tf.variable_scope("cal_accuracy"):
            self.accuracy_pose = self.cal_accuracy(self.gt_poses, self.pd_poses, name="poses_accuracy")

        tf.summary.scalar("pose_accuracy", self.accuracy_pose)
        tf.summary.scalar("pose_loss_scalar", self.pose_loss)
        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()
