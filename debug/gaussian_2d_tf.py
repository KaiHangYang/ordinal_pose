import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
import sys
import time
import cv2

sys.path.append("../")

from utils.preprocess_utils import common

batch_size = 4
nJoints = 17
size = 64

evaluation_iterations = 1000

# input_joints shape (None, 17, 3)
def build_input_2d(input_center, nJoints=17, batch_size=4, size=64, stddev=2.0, name="input_heatmaps"):
    with tf.variable_scope(name):
        raw_arr_y = tf.constant(np.reshape(np.repeat(np.arange(0, size, 1), size), [size, size, 1]).astype(np.float32), name="raw_arr_y")

        arr_y = tf.tile(raw_arr_y[np.newaxis], [batch_size, 1, 1, 1])
        arr_x = tf.tile(tf.transpose(raw_arr_y, perm=[1, 0, 2])[np.newaxis], [batch_size, 1, 1, 1])

        const_x = arr_x
        const_y = arr_y

        all_heatmaps = []
        for j_idx in range(nJoints):
            cur_heatmaps = tf.exp(-(tf.pow(const_x - tf.reshape(input_center[:, j_idx, 0], [-1, 1, 1, 1]), 2) + tf.pow(const_y - tf.reshape(input_center[:, j_idx, 1], [-1, 1, 1, 1]), 2)) / 2.0 / stddev / stddev)
            all_heatmaps.append(cur_heatmaps)

        heatmaps_labels = tf.concat(all_heatmaps, axis=3)

    return heatmaps_labels


if __name__ == "__main__":

    input_batch_size = tf.placeholder(shape=[], dtype=np.int32)
    input_center = tf.placeholder(shape=[None, nJoints, 2], dtype=tf.float32)

    sess = tf.Session()
    with tf.device("/device:GPU:0"):
        output_heatmaps = build_input_2d(input_center, batch_size=input_batch_size)

    gpu_heatmaps = sess.run(output_heatmaps, feed_dict={input_batch_size: 4, input_center: np.random.random([4, nJoints, 2]) * (size - 1)})

    for _ in range(evaluation_iterations):
        batch_size = np.round(np.random.random(1) * 10).astype(np.int32).tolist()[0]
        print(batch_size)

        centers_arr = np.round(np.random.random([batch_size, nJoints, 2]) * (60))
        cpu_heatmaps = np.zeros([batch_size, size, size, nJoints], dtype=np.float32)

        start_time = time.clock()
        for b in range(batch_size):
            for j_idx in range(nJoints):
                cpu_heatmaps[b, :, :, j_idx] = common.make_gaussian(centers_arr[b, j_idx], size=64, ratio=2.0)
        start_time = time.clock() - start_time
        print(start_time)

        start_time = time.clock()
        gpu_heatmaps = sess.run(output_heatmaps, feed_dict={input_center: centers_arr, input_batch_size: batch_size})
        start_time = time.clock() - start_time
        print(start_time)

        # The default channel rank of tensorflow seems stupid here
        # check the gaussian_center
        for b in range(batch_size):
            for j_idx in range(nJoints):
                assert(np.argmax(cpu_heatmaps[b, :, :, j_idx]) == np.argmax(gpu_heatmaps[b, :, :, j_idx]))
                assert(np.argmax(cpu_heatmaps[b, :, :, j_idx]) == size*centers_arr[b, j_idx, 1] + centers_arr[b, j_idx, 0])
                # print(np.max(cpu_heatmaps[b, :, :, j_idx]), np.max(gpu_heatmaps[b, :, :, j_idx]))
                assert(np.max(cpu_heatmaps[b, :, :, j_idx]) == np.max(gpu_heatmaps[b, :, :, j_idx]))
