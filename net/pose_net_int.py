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
    def __init__(self, nJoints, is_training, batch_size, img_size=256, loss_weight_heatmap=1.0, loss_weight_pose=1.0, pose_2d_scale=4.0, pose_3d_scale=1000.0, is_use_bn=True, nFeats=256, nModules=3, nRegModules=2):
        self.model_name = "PoseNet"

        self.loss_weight_heatmap = loss_weight_heatmap
        self.loss_weight_pose = loss_weight_pose
        self.pose_2d_scale = pose_2d_scale
        self.pose_3d_scale = pose_3d_scale

        self.nJoints = nJoints
        self.img_size = img_size
        self.is_use_bias = True
        self.is_tiny = False
        self.is_use_bn = is_use_bn
        self.is_training = is_training
        self.res_utils = mResidualUtils(is_training=self.is_training, is_use_bias=self.is_use_bias, is_tiny=self.is_tiny, is_use_bn=self.is_use_bn)
        self.batch_size = batch_size
        self.feature_size = 64

        self.nModules = nModules
        self.nRegModules = nRegModules
        self.nFeats = nFeats
        self.output_size = 32

    # copy the implementation from https://github.com/geopavlakos/c2f-vol-train/blob/master/src/models/hg-stacked.lua
    def build_model(self, input_images):
        with tf.variable_scope(self.model_name):
            net = mConvBnRelu(inputs=input_images, nOut=64, kernel_size=7, strides=2, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="conv1")

            net = self.res_utils.residual_block(net, 128, name="res1")
            net_pooled = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net_pooled, 128, name="res2")
            net = self.res_utils.residual_block(net, 128, name="res3")
            net = self.res_utils.residual_block(net, self.nFeats, name="res4")

            hg1 = hourglass.build_hourglass(net, self.nFeats, 4, name="hg_1", is_training=self.is_training, res_utils=self.res_utils, nModules=self.nModules)

            lin1 = mConvBnRelu(inputs=hg1, nOut=self.nFeats, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin1")
            lin2 = mConvBnRelu(inputs=lin1, nOut=self.nFeats, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin2")

            self.heatmaps = tf.layers.conv2d(inputs=lin2, filters=self.nJoints, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="heatmaps")
            out1_ = tf.layers.conv2d(inputs=self.heatmaps, filters=self.nFeats+128, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out1_")

            with tf.variable_scope("concat_1"):
                cat1 = tf.concat([lin2, net_pooled], axis=3)
                cat1_ = tf.layers.conv2d(inputs=cat1, filters=self.nFeats+128, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
                int1 = tf.add_n([cat1_, out1_])

            hg2 = hourglass.build_hourglass(int1, self.nFeats, 4, name="hg_2", is_training=self.is_training, res_utils=self.res_utils, nModules=self.nModules)

            reg = hg2
            reg = self.res_utils.residual_block(reg, self.nFeats, name="final_res_1")
            reg = tf.layers.max_pooling2d(reg, pool_size=2, strides=2, padding="VALID", name="final_pool")
            reg = self.res_utils.residual_block(reg, 512, name="final_res_2")

            reg = mConvBnRelu(inputs=reg, nOut=512, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="final_lin1")
            reg = mConvBnRelu(inputs=reg, nOut=self.nJoints * self.output_size, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="final_lin2")


            with tf.variable_scope("integral_pose_block"):
                integral_index = tf.constant(np.arange(0, self.output_size, 1.0).astype(np.float32))
                integral_index = tf.tile(integral_index[tf.newaxis, tf.newaxis], [self.batch_size, self.nJoints, 1]) # shape = (batch_size, nJoints, output_size)
                reg = tf.transpose(tf.reshape(reg, [self.batch_size, self.output_size, self.output_size, self.nJoints, self.output_size]), [0, 3, 1, 2, 4])
                reg = tf.reshape(tf.nn.softmax(tf.reshape(reg, [self.batch_size, self.nJoints, -1]), axis=-1), [self.batch_size, self.nJoints, self.output_size, self.output_size, self.output_size]) # y, x, z axis

                with tf.variable_scope("pose_y"):
                    coord_y = tf.reduce_sum(tf.reduce_sum(reg, axis=[3, 4]) * integral_index, axis=-1) / (self.output_size - 1.0) - 0.5

                with tf.variable_scope("pose_x"):
                    coord_x = tf.reduce_sum(tf.reduce_sum(reg, axis=[2, 4]) * integral_index, axis=-1) / (self.output_size - 1.0) - 0.5

                with tf.variable_scope("pose_z"):
                    coord_z = tf.reduce_sum(tf.reduce_sum(reg, axis=[2, 3]) * integral_index, axis=-1) / (self.output_size - 1.0) - 0.5

                self.poses = tf.concat([coord_x[:, :, tf.newaxis], coord_y[:, :, tf.newaxis], coord_z[:, :, tf.newaxis]], axis=-1)


    def get_joints_hm(self, heatmaps, batch_size, name="heatmap_to_joints"):
        with tf.variable_scope(name):
            with tf.device("/device:GPU:0"):
                max_indices = tf.argmax(tf.reshape(heatmaps, [batch_size, -1, self.nJoints]), axis=1)

            # currently the unravel_index only support cpu.
            with tf.device("cpu:0"):
                cur_joints = tf.unravel_index(tf.reshape(max_indices, [-1]), [self.feature_size, self.feature_size])
            cur_joints = tf.reshape(tf.transpose(cur_joints), [-1, self.nJoints, 2])[:, :, ::-1]

        return tf.cast(cur_joints, tf.float32)

    def cal_accuracy(self, gt_joints, pd_joints, name="accuracy"):
        with tf.variable_scope(name):
            accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(gt_joints - pd_joints, 2), axis=2)))
        return accuracy

    # input_joints shape (None, nJoints, 2)
    def build_input_heatmaps(self, input_center, stddev=1.0, name="input_heatmaps", gaussian_coefficient=False):
        with tf.variable_scope(name):
            raw_arr_y = tf.constant(np.reshape(np.repeat(np.arange(0, self.feature_size, 1), self.feature_size), [self.feature_size, self.feature_size, 1]).astype(np.float32), name="raw_arr_y")

            const_y = tf.tile(raw_arr_y[np.newaxis], [self.batch_size, 1, 1, 1])
            const_x = tf.tile(tf.transpose(raw_arr_y, perm=[1, 0, 2])[np.newaxis], [self.batch_size, 1, 1, 1])

            all_heatmaps = []
            for j_idx in range(self.nJoints):

                if gaussian_coefficient:
                    cur_heatmaps = (1.0 / (2 * np.pi * stddev * stddev)) * tf.exp(-(tf.pow(const_x - tf.reshape(input_center[:, j_idx, 0], [-1, 1, 1, 1]), 2) + tf.pow(const_y - tf.reshape(input_center[:, j_idx, 1], [-1, 1, 1, 1]), 2)) / 2.0 / stddev / stddev)
                else:
                    cur_heatmaps = tf.exp(-(tf.pow(const_x - tf.reshape(input_center[:, j_idx, 0], [-1, 1, 1, 1]), 2) + tf.pow(const_y - tf.reshape(input_center[:, j_idx, 1], [-1, 1, 1, 1]), 2)) / 2.0 / stddev / stddev)

                all_heatmaps.append(cur_heatmaps)

            heatmaps_labels = tf.concat(all_heatmaps, axis=3, name="heatmaps")

        return heatmaps_labels

    def build_evaluation(self):
        self.global_steps = tf.train.get_or_create_global_step()
        with tf.variable_scope("parser_pose"):
            self.pd_poses = (self.poses - tf.tile(self.poses[:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_3d_scale

    def build_loss(self, input_heatmaps, input_poses, lr, lr_decay_step=None, lr_decay_rate=None):
        self.global_steps = tf.train.get_or_create_global_step()

        if lr_decay_step is None or lr_decay_rate is None:
            self.lr = lr
        else:
            self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        with tf.variable_scope("losses"):
            self.heatmap_loss = tf.nn.l2_loss(input_heatmaps - self.heatmaps, name="heatmap_loss") / self.batch_size * self.loss_weight_heatmap
            self.pose_loss = tf.nn.l2_loss(input_poses - self.poses, name="pose_loss") / self.batch_size * self.loss_weight_pose

            self.total_loss = self.heatmap_loss + self.pose_loss

        # NOTICE: The dependencies must be added, because of the BN used in the residual 
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Update_ops num {}".format(len(update_ops)))
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.total_loss, self.global_steps)

        with tf.variable_scope("parser_results"):
            cur_batch_size = tf.cast(self.batch_size, dtype=tf.int32)

            with tf.variable_scope("parser_hm"):
                combined_heatmaps = tf.concat([input_heatmaps, self.heatmaps], axis=0, name="heatmaps_combine")
                all_joints_hm = self.get_joints_hm(combined_heatmaps, batch_size=2*cur_batch_size, name="all_joints_hm")

                self.gt_joints_hm = all_joints_hm[0:cur_batch_size] * self.pose_2d_scale
                self.pd_joints_hm = all_joints_hm[cur_batch_size:cur_batch_size*2] * self.pose_2d_scale

            with tf.variable_scope("parser_pose"):
                self.gt_poses = (input_poses - tf.tile(input_poses[:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_3d_scale
                self.pd_poses = (self.poses - tf.tile(self.poses[:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_3d_scale

        with tf.variable_scope("cal_accuracy"):
            self.accuracy_pose = self.cal_accuracy(self.gt_poses, self.pd_poses, name="poses_accuracy")
            self.accuracy_hm = self.cal_accuracy(self.gt_joints_hm, self.pd_joints_hm, name="hm_joints_accuracy")

        tf.summary.scalar("pose_accuracy", self.accuracy_pose)
        tf.summary.scalar("heatmap_joints_accuracy", self.accuracy_hm)
        tf.summary.scalar("total_loss_scalar", self.total_loss)
        tf.summary.scalar("heatmap_loss_scalar", self.heatmap_loss)
        tf.summary.scalar("pose_loss_scalar", self.pose_loss)
        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()
