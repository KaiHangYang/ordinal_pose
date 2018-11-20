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
    def __init__(self, nJoints, is_training, batch_size, img_size=256, loss_weight_heatmap=1.0, loss_weight_pose=1.0, pose_2d_scale=4.0, pose_3d_scale=1000.0, is_use_bn=True):

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
        self.nModules = 3
        self.nRegModules = 2

    # copy the implementation from https://github.com/geopavlakos/c2f-vol-train/blob/master/src/models/hg-stacked.lua
    def build_model(self, input_images):
        with tf.variable_scope("PoseNet"):
            net = mConvBnRelu(inputs=input_images, nOut=64, kernel_size=7, strides=2, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="conv1")

            net = self.res_utils.residual_block(net, 128, name="res1")
            net_pooled = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net_pooled, 128, name="res2")
            net = self.res_utils.residual_block(net, 128, name="res3")
            net = self.res_utils.residual_block(net, 256, name="res4")

            hg1 = hourglass.build_hourglass(net, 256, 4, name="hg_1", is_training=self.is_training, res_utils=self.res_utils)

            lin1 = mConvBnRelu(inputs=hg1, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin1")
            lin2 = mConvBnRelu(inputs=lin1, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin2")

            self.heatmaps = tf.layers.conv2d(inputs=lin2, filters=self.nJoints, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="heatmaps")
            out1_ = tf.layers.conv2d(inputs=self.heatmaps, filters=256+128, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out1_")

            with tf.variable_scope("concat_1"):
                cat1 = tf.concat([lin2, net_pooled], axis=3)
                cat1_ = tf.layers.conv2d(inputs=cat1, filters=256+128, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
                int1 = tf.add_n([cat1_, out1_])

            hg2 = hourglass.build_hourglass(int1, 256, 4, name="hg_2", is_training=self.is_training, res_utils=self.res_utils)

            lin3 = mConvBnRelu(inputs=hg2, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin3")
            lin4 = mConvBnRelu(inputs=lin3, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, is_use_bn=self.is_use_bn, name="lin4")

            reg = lin4
            with tf.variable_scope("final_res"):
                for i in range(4):
                    reg = self.res_utils.residual_block(reg, 256, name="res{}".format(i))

            # x, y, z, 2d (2d is to get the x, y, z value)
            self.posemaps = tf.layers.conv2d(inputs=reg, filters=self.nJoints*4, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="pose_maps")

    def get_joints_hm(self, heatmaps, batch_size, name="heatmap_to_joints"):
        with tf.variable_scope(name):
            with tf.device("/device:GPU:0"):
                max_indices = tf.argmax(tf.reshape(heatmaps, [batch_size, -1, self.nJoints]), axis=1)

            # currently the unravel_index only support cpu.
            with tf.device("cpu:0"):
                cur_joints = tf.unravel_index(tf.reshape(max_indices, [-1]), [self.feature_size, self.feature_size])
            cur_joints = tf.reshape(tf.transpose(cur_joints), [-1, self.nJoints, 2])[:, :, ::-1]

        return tf.cast(cur_joints, tf.float32)

    # cpu version is faster but here I use it to calculate the accuracy
    def get_joints_xyzm(self, joints_2d, xyzmaps, batch_size, name="xyzmaps_to_joints"):
        with tf.variable_scope(name):
            with tf.device("/device:GPU:0"):
                xmaps = xyzmaps[:, :, :, 0:self.nJoints]
                ymaps = xyzmaps[:, :, :, self.nJoints:2*self.nJoints]
                zmaps = xyzmaps[:, :, :, 2*self.nJoints:3*self.nJoints]

                joints_2d = tf.cast(joints_2d, tf.int32)
                # currently (x, y), then make it (y, x)
                reshaped_joints_2d = tf.reshape(joints_2d, [-1, 2])[:, ::-1]

                reshaped_xmaps = tf.reshape(tf.transpose(xmaps, perm=[0, 3, 1, 2]), [-1, self.feature_size, self.feature_size])
                reshaped_ymaps = tf.reshape(tf.transpose(ymaps, perm=[0, 3, 1, 2]), [-1, self.feature_size, self.feature_size])
                reshaped_zmaps = tf.reshape(tf.transpose(zmaps, perm=[0, 3, 1, 2]), [-1, self.feature_size, self.feature_size])

                value_indices = tf.concat([tf.reshape(tf.range(0, self.nJoints * batch_size), [-1, 1]), reshaped_joints_2d], axis=1)

                x_pos = tf.reshape(tf.gather_nd(reshaped_xmaps, value_indices, name="get_x_pos"), [-1, 1])
                y_pos = tf.reshape(tf.gather_nd(reshaped_ymaps, value_indices, name="get_y_pos"), [-1, 1])
                z_pos = tf.reshape(tf.gather_nd(reshaped_zmaps, value_indices, name="get_z_pos"), [-1, 1])

                joints_3d = tf.reshape(tf.concat([x_pos, y_pos, z_pos], axis=1), [batch_size, self.nJoints, 3])
        return joints_3d

    def build_input_xyzmaps(self, input_joints_3d, name="input_xysmaps"):
        with tf.variable_scope(name):
            joints_x = input_joints_3d[:, :, 0]
            joints_y = input_joints_3d[:, :, 1]
            joints_z = input_joints_3d[:, :, 2]

            xmaps = tf.tile(tf.reshape(joints_x, [self.batch_size, 1, 1, self.nJoints]), [1, self.feature_size, self.feature_size, 1])
            ymaps = tf.tile(tf.reshape(joints_y, [self.batch_size, 1, 1, self.nJoints]), [1, self.feature_size, self.feature_size, 1])
            zmaps = tf.tile(tf.reshape(joints_z, [self.batch_size, 1, 1, self.nJoints]), [1, self.feature_size, self.feature_size, 1])

            xyzmaps = tf.concat([xmaps, ymaps, zmaps], axis=3)

        return xyzmaps

    # input_joints shape (None, nJoints, 2)
    def build_input_heatmaps(self, input_center, stddev=2.0, name="input_heatmaps", gaussian_coefficient=False):
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

    def cal_accuracy(self, gt_joints, pd_joints, name="accuracy"):
        with tf.variable_scope(name):
            accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(gt_joints - pd_joints, 2), axis=2)))
        return accuracy

    def build_evaluation(self):
        self.global_steps = tf.train.get_or_create_global_step()

        with tf.variable_scope("parser_pose"):
            pd_heatmaps = self.posemaps[:, :, :, 0:self.nJoints]
            pd_xyzmaps = self.posemaps[:, :, :, self.nJoints:]

            pd_hm_joints = self.get_joints_hm(pd_heatmaps, batch_size=self.batch_size, name="hm_joints")
            pd_xyz_joints = self.get_joints_xyzm(pd_hm_joints, pd_xyzmaps, batch_size=self.batch_size, name="xyz_joints")

            self.pd_2d = pd_hm_joints * self.pose_2d_scale
            self.pd_3d = (pd_xyz_joints - tf.tile(pd_xyz_joints[:, 0][:, tf.newaxis], [1, self.nJoints, 1])) * self.pose_3d_scale

    def build_loss(self, input_heatmaps, input_xyzmaps, lr, lr_decay_step, lr_decay_rate):

        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        with tf.variable_scope("losses"):
            self.heatmap_loss = tf.nn.l2_loss(input_heatmaps - self.heatmaps, name="heatmap_loss") / self.batch_size * self.loss_weight_heatmap
            # include 2 part( heatmaps + xyzmaps )
            with tf.variable_scope("posemap_loss"):
                self.posemap_loss = tf.nn.l2_loss(input_heatmaps - self.posemaps[:, :, :, 0:self.nJoints], name="posemap_hm_loss")

                repeated_heatmaps = tf.tile(input_heatmaps, [1, 1, 1, 3])
                self.posemap_loss = self.posemap_loss + tf.nn.l2_loss(repeated_heatmaps * (input_xyzmaps - self.posemaps[:, :, :, self.nJoints:]), name="posemap_xyzm_loss")
                self.posemap_loss = self.posemap_loss / self.batch_size * self.loss_weight_pose

            self.total_loss = self.heatmap_loss + self.posemap_loss

        # NOTICE: The dependencies must be added, because of the BN used in the residual 
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Update_ops num {}".format(len(update_ops)))
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.total_loss, self.global_steps)

        with tf.variable_scope("parser_results"):
            cur_batch_size = tf.cast(self.batch_size, dtype=tf.int32)

            with tf.variable_scope("parser_hms"):
                combined_heatmaps = tf.concat([input_heatmaps, self.heatmaps, self.posemaps[:, :, :, 0:self.nJoints]], axis=0, name="heatmaps_combine")
                all_joints_hm = self.get_joints_hm(combined_heatmaps, batch_size=3*cur_batch_size, name="all_joints_hm")

                gt_joints_2d = all_joints_hm[0:cur_batch_size]
                pd_joints_2d_mid = all_joints_hm[cur_batch_size:cur_batch_size*2]
                pd_joints_2d_final = all_joints_hm[2*cur_batch_size:]

            with tf.variable_scope("parser_pose"):
                combined_xyzmaps = tf.concat([input_xyzmaps, self.posemaps[:, :, :, self.nJoints:]], axis=0, name="xyzmaps_combine")
                all_xyz_joints = self.get_joints_xyzm(pd_joints_2d_final, combined_xyzmaps, batch_size=2*cur_batch_size, name="all_xyz_joints")

                gt_joints_3d = all_xyz_joints[0:cur_batch_size]
                gt_joints_3d = gt_joints_3d - tf.tile(gt_joints_3d[:, 0][:, tf.newaxis], [1, self.nJoints, 1])

                pd_joints_3d = all_xyz_joints[cur_batch_size:]
                pd_joints_3d = pd_joints_3d - tf.tile(pd_joints_3d[:, 0][:, tf.newaxis], [1, self.nJoints, 1])

            with tf.variable_scope("recover"):
                self.gt_2d = gt_joints_2d * self.pose_2d_scale
                self.pd_2d_mid = pd_joints_2d_mid * self.pose_2d_scale
                self.pd_2d_final = pd_joints_2d_final * self.pose_2d_scale

                self.gt_3d = gt_joints_3d * self.pose_3d_scale
                self.pd_3d = pd_joints_3d * self.pose_3d_scale

        with tf.variable_scope("cal_accuracy"):
            self.accuracy_3d = self.cal_accuracy(self.gt_3d, self.pd_3d, name="final_3d_accuracy")
            self.accuracy_2d_mid = self.cal_accuracy(self.gt_2d, self.pd_2d_mid, name="mid_2d_accuracy")
            self.accuracy_2d_final = self.cal_accuracy(self.gt_2d, self.pd_2d_final, name="final_2d_accuracy")

        tf.summary.scalar("final_3d_accuracy", self.accuracy_3d)
        tf.summary.scalar("mid_2d_accuracy", self.accuracy_2d_mid)
        tf.summary.scalar("final_2d_accuracy", self.accuracy_2d_final)

        tf.summary.scalar("total_loss_scalar", self.total_loss)
        tf.summary.scalar("heatmap_loss_scalar", self.heatmap_loss)
        tf.summary.scalar("posemap_loss_scalar", self.posemap_loss)
        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()
