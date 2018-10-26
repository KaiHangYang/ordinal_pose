import os
import sys
import numpy as np
import tensorflow as tf

from network_utils import mResidualUtils
from network_utils import mConvBnRelu
import hourglass

# The structure is translate from github.com/umich-vl/pose-hg-train/blob/maskter/src/models/hg.lua

# is_training is a tensor or python bool
class mOrdinal_F(object):
    def __init__(self, nJoints, is_training, batch_size, img_size=256, loss_weight_heatmap=1.0, loss_weight_volume=1.0, loss_weight_rank=1.0, loss_weight_2d=1000.0):

        self.loss_weight_heatmap = loss_weight_heatmap
        self.loss_weight_volume = loss_weight_volume

        # Loss weight for no gt training
        self.loss_weight_rank = loss_weight_rank
        self.loss_weight_2d = loss_weight_2d

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
        with tf.variable_scope("ordinal_F"):
            net = mConvBnRelu(inputs=input_images, nOut=64, kernel_size=7, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="conv1")

            net = self.res_utils.residual_block(net, 128, name="res1")
            net_pooled = net
            # net_pooled = tf.layers.max_pooling2d(net, 2, 2, name="pooling")
            net = self.res_utils.residual_block(net_pooled, 128, name="res2")
            net = self.res_utils.residual_block(net, 128, name="res3")
            net = self.res_utils.residual_block(net, 256, name="res4")

            hg1 = hourglass.build_hourglass(net, 512, 4, name="hg_1", is_training=self.is_training, res_utils=self.res_utils)

            lin1 = mConvBnRelu(inputs=hg1, nOut=512, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="lin1")
            lin2 = mConvBnRelu(inputs=lin1, nOut=256, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="lin2")

            self.heatmaps = tf.layers.conv2d(inputs=lin2, filters=self.nJoints, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="heatmaps")
            out1_ = tf.layers.conv2d(inputs=self.heatmaps, filters=256+128, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="out1_")

            with tf.variable_scope("concat_1"):
                cat1 = tf.concat([lin2, net_pooled], axis=3)
                cat1_ = tf.layers.conv2d(inputs=cat1, filters=256+128, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
                int1 = tf.add_n([cat1_, out1_])

            hg2 = hourglass.build_hourglass(int1, 512, 4, name="hg_2", is_training=self.is_training, res_utils=self.res_utils)

            lin3 = mConvBnRelu(inputs=hg2, nOut=512, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="lin3")
            lin4 = mConvBnRelu(inputs=lin3, nOut=512, kernel_size=1, strides=1, is_use_bias=self.is_use_bias, is_training=self.is_training, name="lin4")

            self.volumes = tf.layers.conv2d(inputs=lin4, filters=self.nJoints*self.feature_size, kernel_size=1, strides=1, use_bias=self.is_use_bias, padding="SAME", activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="volumes")

    def get_joints_hm(self, heatmaps, batch_size, name="heatmap_to_joints"):
        with tf.variable_scope(name):
            with tf.device("/device:GPU:0"):
                max_indices = tf.argmax(tf.reshape(heatmaps, [batch_size, -1, self.nJoints]), axis=1)

            # currently the unravel_index only support cpu.
            with tf.device("cpu:0"):
                cur_joints = tf.unravel_index(tf.reshape(max_indices, [-1]), [self.feature_size, self.feature_size])
            cur_joints = tf.reshape(tf.transpose(cur_joints), [-1, self.nJoints, 2])[:, :, ::-1]

        return tf.cast(cur_joints, tf.float32)

    def get_joints_vol(self, volumes, batch_size, name="volume_to_joints"):
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

    def cal_accuracy(self, gt_joints, pd_joints, name="accuracy"):
        with tf.variable_scope(name):
            accuracy = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(gt_joints - pd_joints, 2), axis=2)))
        return accuracy

    # input_joints shape (None, 17, 2)
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

    # input_joints shape (None, 17, 3)
    def build_input_volumes(self, input_centers, stddev=2.0, name="input_volumes", gaussian_coefficient=False):
        with tf.variable_scope(name):
            raw_arr_y = tf.constant(np.reshape(np.repeat(np.arange(0, self.feature_size, 1), self.feature_size*self.feature_size), [self.feature_size, self.feature_size, self.feature_size]).astype(np.float32), name="raw_arr_y")

            const_y = tf.tile(raw_arr_y[np.newaxis], [self.batch_size, 1, 1, 1], name="const_y")
            const_x = tf.tile(tf.transpose(raw_arr_y, perm=[1, 0, 2])[np.newaxis], [self.batch_size, 1, 1, 1], name="const_x")
            const_z = tf.tile(tf.transpose(raw_arr_y, perm=[2, 1, 0])[np.newaxis], [self.batch_size, 1, 1, 1], name="const_z")

            all_vols = []
            for j_idx in range(self.nJoints):
                if gaussian_coefficient:
                    cur_vol =  (1.0 / (2 * np.pi * stddev * stddev)) * tf.exp(-(tf.pow(const_x - tf.reshape(input_centers[:, j_idx, 0], [-1, 1, 1, 1]), 2) + tf.pow(const_y - tf.reshape(input_centers[:, j_idx, 1], [-1, 1, 1, 1]), 2) + tf.pow(const_z - tf.reshape(input_centers[:, j_idx, 2], [-1, 1, 1, 1]), 2)) / 2.0 / stddev / stddev)
                else:
                    cur_vol = tf.exp(-(tf.pow(const_x - tf.reshape(input_centers[:, j_idx, 0], [-1, 1, 1, 1]), 2) + tf.pow(const_y - tf.reshape(input_centers[:, j_idx, 1], [-1, 1, 1, 1]), 2) + tf.pow(const_z - tf.reshape(input_centers[:, j_idx, 2], [-1, 1, 1, 1]), 2)) / 2.0 / stddev / stddev)
                all_vols.append(cur_vol)

            vol_labels = tf.concat(all_vols, axis=3, name="volumes")

        return vol_labels

    def flip_volumes(self, vols, flip_array, name="flip_volume"):
        with tf.variable_scope(name):
            # flip the features map first
            flipped_vols = tf.image.flip_left_right(vols)
            # get the new vol orders
            old_order = np.arange(0, self.nJoints, 1)
            cur_order = old_order.copy()

            cur_order[flip_array[:, 0]] = old_order[flip_array[:, 1]]
            cur_order[flip_array[:, 1]] = old_order[flip_array[:, 0]]

            cur_vols = []
            for j in range(self.nJoints):
                cur_idx = cur_order[j]
                cur_vols.append(flipped_vols[:, :, :, self.feature_size*cur_idx:self.feature_size*(cur_idx+1)])

            return tf.concat(cur_vols, axis=3, name="flipped_vols")

    def build_evaluation_nogt(self, eval_batch_size, flip_array):
        self.global_steps = tf.train.get_or_create_global_step()
        # The self.volumes contains [raw_img_outputs, flipped_img_outputs]
        with tf.variable_scope("parser_joints"):
            raw_volumes = self.volumes[0:eval_batch_size]
            flipped_volumes = self.flip_volumes(self.volumes[eval_batch_size:2*eval_batch_size], flip_array=flip_array)
            # mean_volumes = (raw_volumes + flipped_vol_batchs) / 2.0

            with tf.variable_scope("data_decompositon"):
                volumes_z_indices = tf.tile(np.arange(0.0, self.feature_size, 1.0).astype(np.float32)[np.newaxis, :, np.newaxis], [eval_batch_size, 1, self.nJoints])

                flipped_reshaped_volumes = tf.transpose(tf.reshape(flipped_volumes, [-1, self.feature_size, self.feature_size, self.nJoints, self.feature_size]), perm=[0, 1, 2, 4, 3])
                flipped_softmaxed_volumes = tf.reshape(tf.nn.softmax(tf.reshape(flipped_reshaped_volumes, [eval_batch_size, -1, self.nJoints]), axis=1), [eval_batch_size, self.feature_size, self.feature_size, self.feature_size, self.nJoints])
                flipped_volumes_xy = tf.reduce_sum(flipped_softmaxed_volumes, axis=[3])
                flipped_volumes_z_arrs = tf.reduce_sum(flipped_softmaxed_volumes, axis=[1, 2])
                self.flipped_volumes_z = tf.reduce_sum(flipped_volumes_z_arrs * volumes_z_indices, axis=1)

                raw_reshaped_volumes = tf.transpose(tf.reshape(raw_volumes, [-1, self.feature_size, self.feature_size, self.nJoints, self.feature_size]), perm=[0, 1, 2, 4, 3])
                raw_softmaxed_volumes = tf.reshape(tf.nn.softmax(tf.reshape(raw_reshaped_volumes, [eval_batch_size, -1, self.nJoints]), axis=1), [eval_batch_size, self.feature_size, self.feature_size, self.feature_size, self.nJoints])
                raw_volumes_xy = tf.reduce_sum(raw_softmaxed_volumes, axis=[3])
                raw_volumes_z_arrs = tf.reduce_sum(raw_softmaxed_volumes, axis=[1, 2])
                self.raw_volumes_z = tf.reduce_sum(raw_volumes_z_arrs * volumes_z_indices, axis=1)

            combined_heatmaps = tf.concat([flipped_volumes_xy, raw_volumes_xy], axis=0)
            all_joints_2d = self.get_joints_hm(combined_heatmaps, batch_size=2*eval_batch_size, name="all_joints_2d")

            self.flipped_joints_2d = all_joints_2d[0:eval_batch_size]
            self.raw_joints_2d = all_joints_2d[eval_batch_size:2*eval_batch_size]

            self.mean_joints_2d = (self.flipped_joints_2d + self.raw_joints_2d) / 2.0
            self.mean_volumes_z = (self.raw_volumes_z + self.flipped_volumes_z) / 2.0

    # eval_batch_size mean the real batch_size(normally 0.5 * self.batch_size)
    # input_volumes
    def build_evaluation(self, eval_batch_size, flip_array):
        self.global_steps = tf.train.get_or_create_global_step()
        # The self.volumes contains [raw_img_outputs, flipped_img_outputs]
        with tf.variable_scope("parser_joints"):
            raw_vol_batchs = self.volumes[0:eval_batch_size]
            flipped_vol_batchs = self.flip_volumes(self.volumes[eval_batch_size:2*eval_batch_size], flip_array=flip_array)
            mean_volumes = (raw_vol_batchs + flipped_vol_batchs) / 2.0

            combined_volumes = tf.concat([mean_volumes, raw_vol_batchs], axis=0, name="combined_volumes")
            all_joints = self.get_joints_vol(combined_volumes, batch_size=2*eval_batch_size, name="all_joints")

            self.mean_joints = all_joints[0:eval_batch_size]
            self.raw_joints = all_joints[eval_batch_size:2*eval_batch_size]

    # ordinal_F with ground true volumes
    def build_loss_gt(self, input_heatmaps, input_volumes, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase= True, name= 'learning_rate')

        with tf.variable_scope("losses"):
            self.heatmap_loss = tf.nn.l2_loss(input_heatmaps - self.heatmaps, name="heatmap_loss") / self.batch_size * self.loss_weight_heatmap
            self.volume_loss = tf.nn.l2_loss(input_volumes - self.volumes, name="volume_loss") / self.batch_size * self.loss_weight_volume

            self.total_loss = self.heatmap_loss + self.volume_loss

        # NOTICE: The dependencies must be added, because of the BN used in the residual 
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Update_ops num {}".format(len(update_ops)))
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.total_loss, self.global_steps)

        with tf.variable_scope("parser_joints"):
            cur_batch_size = tf.cast(self.batch_size, dtype=tf.int32)

            with tf.variable_scope("parser_hm"):
                combined_heatmaps = tf.concat([input_heatmaps, self.heatmaps], axis=0, name="heatmaps_combine")
                all_joints_hm = self.get_joints_hm(combined_heatmaps, batch_size=2*cur_batch_size, name="all_joints_hm")

                self.gt_joints_hm = all_joints_hm[0:cur_batch_size]
                self.pd_joints_hm = all_joints_hm[cur_batch_size:cur_batch_size*2]

            with tf.variable_scope("parser_vol"):
                combined_volumes = tf.concat([input_volumes, self.volumes], axis=0, name="volumes_combine")
                all_joints_vol  = self.get_joints_vol(combined_volumes, batch_size=2*cur_batch_size, name="all_joints_vol")

                self.gt_joints_vol = all_joints_vol[0:cur_batch_size]
                self.pd_joints_vol = all_joints_vol[cur_batch_size:cur_batch_size*2]

        with tf.variable_scope("cal_accuracy"):
            self.accuracy_vol = self.cal_accuracy(self.gt_joints_vol, self.pd_joints_vol, name="vol_joints_accuracy")
            self.accuracy_hm = self.cal_accuracy(self.gt_joints_hm, self.pd_joints_hm, name="hm_joints_accuracy")

        tf.summary.scalar("volume_joints_accuracy", self.accuracy_vol)
        tf.summary.scalar("heatmap_joints_accuracy", self.accuracy_hm)

        tf.summary.scalar("total_loss_scalar", self.total_loss)
        tf.summary.scalar("heatmap_loss_scalar", self.heatmap_loss)
        tf.summary.scalar("volume_loss_scalar", self.volume_loss)
        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()

    def build_loss_no_gt(self, input_heatmaps, relation_table, loss_table_log, loss_table_pow, lr, lr_decay_step, lr_decay_rate):
        self.global_steps = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(learning_rate=lr, global_step=self.global_steps, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase=True, name="learning_rate")

        with tf.variable_scope("losses"):
            with tf.variable_scope("hm_loss"):
                self.hm_loss = tf.nn.l2_loss(self.heatmaps - input_heatmaps) / self.batch_size * self.loss_weight_heatmap

            with tf.variable_scope("vol_loss"):
                self.vol_loss = 0
                with tf.variable_scope("data_decomposition"):
                    reshaped_volumes = tf.transpose(tf.reshape(self.volumes, [-1, self.feature_size, self.feature_size, self.nJoints, self.feature_size]), perm=[0, 1, 2, 4, 3])
                    softmaxed_volumes = tf.reshape(tf.nn.softmax(tf.reshape(reshaped_volumes, [self.batch_size, -1, self.nJoints]), axis=1), [self.batch_size, self.feature_size, self.feature_size, self.feature_size, self.nJoints])

                    volumes_xy = tf.reduce_sum(softmaxed_volumes, axis=[3])
                    volumes_z_arrs = tf.reduce_sum(softmaxed_volumes, axis=[1, 2])
                    volumes_z_indices = tf.tile(np.arange(0.0, self.feature_size, 1.0).astype(np.float32)[np.newaxis, :, np.newaxis], [self.batch_size, 1, self.nJoints])

                    volumes_z = tf.reduce_sum(volumes_z_arrs * volumes_z_indices, axis=1)

                with tf.variable_scope("loss_rank"):
                    row_val = tf.tile(volumes_z[:, :, tf.newaxis], [1, 1, self.nJoints])
                    col_val = tf.tile(volumes_z[:, tf.newaxis], [1, self.nJoints, 1])

                    rel_distance = (row_val - col_val)
                    # Softplus is log(1 + exp(x)) and without overflow
                    self.loss_rank = tf.reduce_sum(loss_table_log * tf.math.softplus(relation_table * rel_distance) + loss_table_pow * tf.pow(rel_distance, 2)) / self.batch_size * self.loss_weight_rank

                with tf.variable_scope("2d_loss"):
                    self.loss_2d = tf.nn.l2_loss(volumes_xy - input_heatmaps) / self.batch_size * self.loss_weight_2d

                self.vol_loss = self.loss_rank + self.loss_2d

            self.total_loss = self.vol_loss + self.hm_loss

        # NOTICE: The dependencies must be added, because of the BN used in the residual 
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Update_ops num {}".format(len(update_ops)))
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.total_loss, self.global_steps)

        with tf.variable_scope("parser_joints"):
            cur_batch_size = tf.cast(self.batch_size, dtype=tf.int32)

            with tf.variable_scope("parser_hm"):
                combined_heatmaps = tf.concat([input_heatmaps, volumes_xy], axis=0, name="heatmaps_combine")
                all_joints_hm = self.get_joints_hm(combined_heatmaps, batch_size=2*cur_batch_size, name="all_joints_hm")

                self.gt_joints_hm = all_joints_hm[0:cur_batch_size]
                self.pd_joints_hm = all_joints_hm[cur_batch_size:cur_batch_size*2]

        with tf.variable_scope("cal_accuracy"):
            self.accuracy_hm = self.cal_accuracy(self.gt_joints_hm, self.pd_joints_hm, name="hm_joints_accuracy")

        tf.summary.scalar("heatmap_joints_accuracy", self.accuracy_hm)
        tf.summary.scalar("loss_hm", self.hm_loss)
        tf.summary.scalar("vol_loss", self.vol_loss)
        tf.summary.scalar("loss_rank", self.loss_rank)
        tf.summary.scalar("loss_2d", self.loss_2d)
        tf.summary.scalar("learning_rate", self.lr)

        self.merged_summary = tf.summary.merge_all()
