import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import sys
import tensorflow as tf
import cv2
import time

sys.path.append("../")
from utils.dataread_utils import ordinal_3_1_reader as ordinal_reader
from utils.preprocess_utils import ordinal_3_1 as preprocessor
from utils.visualize_utils import display_utils

import configs
# t means gt(0) or ord(1)
# sec is 0:"3_1" or 1:"3_2" or 2:"3_3"
configs.parse_configs(1, 0)
configs.print_configs()

depth_scale = 1.0

def manual_cal_rank_loss(depths, scale):
    total_loss = 0
    for i in range(len(depths)):
        for j in range(i+1, len(depths)):
            if scale * np.abs(depths[i] - depths[j]) < 100:
                total_loss += (depths[i] - depths[j]) ** 2
            elif depths[i] < depths[j]:
                total_loss += np.log(1 + np.exp(depths[i] - depths[j]))
            else:
                total_loss += np.log(1 + np.exp(- depths[i] + depths[j]))
    return total_loss

if __name__ == "__main__":

    ############################ Train and valid data list ##########################
    train_range = np.load(configs.train_range_file)
    np.random.shuffle(train_range)

    valid_range = np.load(configs.valid_range_file)
    train_img_list = [configs.train_img_path_fn(i) for i in train_range]
    train_lbl_list = [configs.train_lbl_path_fn(i) for i in train_range]

    valid_img_list = [configs.valid_img_path_fn(i) for i in valid_range]
    valid_lbl_list = [configs.valid_lbl_path_fn(i) for i in valid_range]
    ###################################################################

    with tf.device('/cpu:0'):
        train_data_iter, train_data_init_op = ordinal_reader.get_data_iterator(train_img_list, train_lbl_list, batch_size=1, name="train_reader")
        valid_data_iter, valid_data_init_op = ordinal_reader.get_data_iterator(valid_img_list, valid_lbl_list, batch_size=1, name="valid_reader", is_shuffle=False)

    depths = tf.placeholder(shape=[1, configs.nJoints], dtype=tf.float32, name="depths")
    relation_table = tf.placeholder(shape=[1, configs.nJoints, configs.nJoints], dtype=tf.float32, name="relation_table")
    loss_table_log = tf.placeholder(shape=[1, configs.nJoints, configs.nJoints], dtype=tf.float32, name="loss_table_log")
    loss_table_pow = tf.placeholder(shape=[1, configs.nJoints, configs.nJoints], dtype=tf.float32, name="loss_table_pow")

    row_val = tf.tile(depths[:, :, tf.newaxis], [1, 1, configs.nJoints])
    col_val = tf.tile(depths[:, tf.newaxis], [1, configs.nJoints, 1])

    rel_distance = (row_val - col_val)
    result_loss_log = tf.reduce_sum(loss_table_log * tf.log(1 + tf.exp(relation_table * rel_distance)))
    result_loss_pow = tf.reduce_sum(loss_table_pow * tf.pow(rel_distance, 2))
    total_loss = result_loss_log + result_loss_pow

    is_valid = False
    counter = 0
    with tf.Session() as sess:
        sess.run([train_data_init_op, valid_data_init_op])

        while True:
            # get the data path
            if is_valid:
                cur_data_batch = sess.run(valid_data_iter)
            else:
                cur_data_batch = sess.run(train_data_iter)

            batch_size = len(cur_data_batch[0])
            batch_depth_np = np.zeros([batch_size, configs.nJoints], dtype=np.float32)
            batch_relation_table_np = np.zeros([batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)
            batch_loss_table_log_np = np.zeros([batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)
            batch_loss_table_pow_np = np.zeros([batch_size, configs.nJoints, configs.nJoints], dtype=np.float32)

            for b in range(batch_size):
                cur_label = np.load(cur_data_batch[1][b]).tolist()
                cur_joints = np.concatenate([cur_label["joints_2d"], cur_label["joints_3d"][:, 2][:, np.newaxis]], axis=1)

                # Cause the dataset is to large, test no augment
                # cur_img, cur_joints, is_do_flip = preprocessor.preprocess(cur_img, cur_joints)

                batch_depth_np[b] = (cur_joints[:, 2] - cur_joints[0, 2]) / depth_scale
                batch_relation_table_np[b], batch_loss_table_log_np[b], batch_loss_table_pow_np[b] = preprocessor.get_relation_table(cur_joints[:, 2])

            my_rank_loss_tf = sess.run(total_loss, feed_dict={depths: batch_depth_np, relation_table: batch_relation_table_np, loss_table_log: batch_loss_table_log_np, loss_table_pow: batch_loss_table_pow_np})
            manual_rank_loss = manual_cal_rank_loss(batch_depth_np[0], depth_scale)

            print(counter)
            assert(np.abs(manual_rank_loss - my_rank_loss_tf) < 1)
            counter += 1

