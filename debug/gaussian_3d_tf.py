import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import tensorflow as tf
import sys
import time

sys.path.append("../")

from utils.preprocess_utils import common

nJoints = 17

if __name__ == "__main__":
    start_time = time.clock()
    centers_arr = np.random.random([nJoints, 3]) * 3

    cpu_vols = np.zeros([nJoints, 64, 64, 64])
    for i in range(nJoints):
        vol = common.make_gaussian_3d(63 * np.random.random([3]), 64)
    start_time = time.clock() - start_time
    print(start_time)

    input_center = tf.placeholder(shape=[nJoints, 3], dtype=tf.float32)
    raw_arr_y = np.reshape(np.repeat(np.arange(0, 64, 1), 64*64), [64, 64, 64])
    arr_y = np.repeat(raw_arr_y[np.newaxis], nJoints, axis=0)
    arr_x = np.repeat(np.transpose(raw_arr_y, [1, 0, 2])[np.newaxis], nJoints, axis=0)
    arr_z = np.repeat(np.transpose(raw_arr_y, [2, 1, 0])[np.newaxis], nJoints, axis=0)

    sess = tf.Session()
    with tf.device("/device:GPU:0"):
        const_x = tf.constant(arr_x, dtype=tf.float32)
        const_y = tf.constant(arr_y, dtype=tf.float32)
        const_z = tf.constant(arr_z, dtype=tf.float32)

        input_center_reshaped = tf.reshape(input_center, [nJoints, 1, 1, 1, -1])

        output_vol = tf.exp(-(tf.pow(const_x - input_center_reshaped[:, :, :, :, 0], 2) + tf.pow(const_y - input_center_reshaped[:, :, :, :, 1], 2) + tf.pow(const_z - input_center_reshaped[:, :, :, :, 2], 2)) / 2.0 / 2.0 / 2.0)

    sess.run(output_vol, feed_dict={input_center: np.random.random([nJoints, 3]) * 3})

    start_time = time.clock()
    vol = sess.run(output_vol, feed_dict={input_center: centers_arr})
    start_time = time.clock() - start_time
    print(start_time)
