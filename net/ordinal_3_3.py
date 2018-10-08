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
        self.feature_size = 64

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

            self.volumes = tf.layers.conv2d(inputs=lin2, filters=self.nJoints*self.feature_size, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="volumes")

    # new fast version 40ms for batch_size 4
    def get_joints(self, volumes, batch_size, name="volume_to_joints"):
        # volume shape (batch_size, feature_size, feature_size, feature_size * nJoints)
        with tf.device("/device:GPU:0"):
            with tf.variable_scope(name):
                cur_volumes = tf.reshape(tf.transpose(tf.reshape(volumes, [batch_size, self.feature_size, self.feature_size, self.nJoints, self.feature_size]), perm=[0, 1, 2, 4, 3]), [batch_size, -1, self.nJoints])
                cur_argmax_index = tf.reshape(tf.argmax(cur_volumes, axis=1), [-1])

                with tf.device("/cpu:0"):
                    cur_joints = tf.unravel_index(cur_argmax_index, [self.feature_size, self.feature_size, self.feature_size])

                cur_joints = tf.reshape(tf.transpose(cur_joints), [-1, self.nJoints, 3])
                cur_joints = tf.concat([cur_joints[:, :, 0:2][:, :, ::-1], cur_joints[:, :, 2][:, :, tf.newaxis]], axis=2)
                return tf.cast(cur_joints, tf.float32)

    # 100ms for batch_size 4
    # def get_joints(self, volumes, name="volume_to_joints"):
        # # volume shape (batch_size, feature_size, feature_size, feature_size * nJoints)
        # with tf.device("/device:GPU:0"):
            # with tf.variable_scope(name):
                # all_joints = []
                # for i in range(self.nJoints):
                    # cur_volume = volumes[:, :, :, self.feature_size*i:self.feature_size*(i+1)]
                    # cur_argmax_index = tf.argmax(tf.layers.flatten(cur_volume), axis=1)

                    # with tf.device("cpu:0"):
                        # cur_joints = tf.transpose(tf.unravel_index(cur_argmax_index, [self.feature_size, self.feature_size, self.feature_size]))[:, np.newaxis]
                    # all_joints.append(tf.concat([cur_joints[:, :, 0:2][:, :, ::-1], cur_joints[:, :, 2][:, :, np.newaxis]], axis=2))

                # return tf.cast(tf.concat(all_joints, axis=1), tf.float32)

    def cal_accuracy(self, gt_joints, pd_joints):
        accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(gt_joints - pd_joints, 2), axis=2)))
        return accuracy

    # input_joints shape (None, 17, 3)
    def build_input_volumes(self, input_centers, stddev=2.0, name="input_vols"):
        with tf.variable_scope(name):
            raw_arr_y = tf.constant(np.reshape(np.repeat(np.arange(0, self.feature_size, 1), self.feature_size*self.feature_size), [self.feature_size, self.feature_size, self.feature_size]).astype(np.float32), name="raw_arr_y")

            const_y = tf.tile(raw_arr_y[np.newaxis], [self.batch_size, 1, 1, 1], name="const_y")
            const_x = tf.tile(tf.transpose(raw_arr_y, perm=[1, 0, 2])[np.newaxis], [self.batch_size, 1, 1, 1], name="const_x")
            const_z = tf.tile(tf.transpose(raw_arr_y, perm=[2, 1, 0])[np.newaxis], [self.batch_size, 1, 1, 1], name="const_z")

            all_vols = []
            for j_idx in range(self.nJoints):
                # cur_vol =  (1.0 / (2 * np.pi * stddev * stddev)) * tf.exp(-(tf.pow(const_x - tf.reshape(input_centers[:, j_idx, 0], [-1, 1, 1, 1]), 2) + tf.pow(const_y - tf.reshape(input_centers[:, j_idx, 1], [-1, 1, 1, 1]), 2) + tf.pow(const_z - tf.reshape(input_centers[:, j_idx, 2], [-1, 1, 1, 1]), 2)) / 2.0 / stddev / stddev)
                cur_vol = tf.exp(-(tf.pow(const_x - tf.reshape(input_centers[:, j_idx, 0], [-1, 1, 1, 1]), 2) + tf.pow(const_y - tf.reshape(input_centers[:, j_idx, 1], [-1, 1, 1, 1]), 2) + tf.pow(const_z - tf.reshape(input_centers[:, j_idx, 2], [-1, 1, 1, 1]), 2)) / 2.0 / stddev / stddev)
                all_vols.append(cur_vol)

            vol_labels = tf.concat(all_vols, axis=3, name="volumes")

        return vol_labels

    # ordinal_3_3 with ground true volumes
    def build_loss_gt(self, input_volumes, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        with tf.variable_scope("loss"):
            self.loss = tf.nn.l2_loss(self.volumes - input_volumes, name="volume_loss") / self.batch_size * self.loss_weight_volume

        # NOTICE: The dependencies must be added, because of the BN used in the residual 
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Update_ops num {}".format(len(update_ops)))
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, self.global_steps)

        with tf.variable_scope("parser_joints"):
            combined_volumes = tf.concat([input_volumes, self.volumes], axis=0, name="volume_combine")
            cur_batch_size = tf.cast(self.batch_size, dtype=tf.int32)
            all_joints  = self.get_joints(combined_volumes, batch_size=2*cur_batch_size, name="all_joints")

            self.gt_joints = all_joints[0:cur_batch_size]
            self.pd_joints = all_joints[cur_batch_size:cur_batch_size*2]

        with tf.variable_scope("cal_accuracy"):
            self.accuracy = self.cal_accuracy(self.gt_joints, self.pd_joints)

        tf.summary.scalar("volume_joints_accuracy", self.accuracy)
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
