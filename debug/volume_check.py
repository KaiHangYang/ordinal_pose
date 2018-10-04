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
from utils.postprocess_utils import volume_utils

import configs
# t means gt(0) or ord(1)
# sec is 0:"3_1" or 1:"3_2" or 2:"3_3"
configs.parse_configs(1, 2)
configs.print_configs()

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

    is_valid = True

    with tf.Session() as sess:
        sess.run([train_data_init_op, valid_data_init_op])

        while True:
            # get the data path
            if is_valid:
                cur_data_batch = sess.run(valid_data_iter)
            else:
                cur_data_batch = sess.run(train_data_iter)

            cur_label = np.load(cur_data_batch[1][0]).tolist()

            cur_joints_zidx = cur_label["joints_zidx"].copy() - 1
            cur_joints = np.concatenate([cur_label["joints_2d"] / 4, cur_joints_zidx[:, np.newaxis]], axis=1)

            all_vols = np.zeros([1, 64, 64, 64*17], dtype=np.float32)

            for j_idx in range(configs.nJoints):
                all_vols[:, :, :, 64*j_idx:64*(j_idx+1)] = preprocessor.make_gaussian_3d(cur_joints[j_idx], size=64, ratio=2)
            vol_joints = volume_utils.get_joints_from_volume(all_vols, volume_size=64, nJoints=17)

            np.max(np.abs(cur_joints - vol_joints))
