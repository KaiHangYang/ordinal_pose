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
def build_input_3d(input_center, nJoints=17, batch_size=4, size=64, stddev=2.0, name="input_vols"):
    with tf.variable_scope(name):
        raw_arr_y = np.reshape(np.repeat(np.arange(0, size, 1), size*size), [size, size, size]).astype(np.float32)

        arr_y = np.repeat(raw_arr_y[np.newaxis], batch_size, axis=0)
        arr_x = np.repeat(np.transpose(raw_arr_y, [1, 0, 2])[np.newaxis], batch_size, axis=0)
        arr_z = np.repeat(np.transpose(raw_arr_y, [2, 1, 0])[np.newaxis], batch_size, axis=0)

        const_x = tf.constant(arr_x, dtype=tf.float32)
        const_y = tf.constant(arr_y, dtype=tf.float32)
        const_z = tf.constant(arr_z, dtype=tf.float32)

        all_vols = []
        for j_idx in range(nJoints):
            cur_vol = tf.exp(-(tf.pow(const_x - tf.reshape(input_center[:, j_idx, 0], [-1, 1, 1, 1]), 2) + tf.pow(const_y - tf.reshape(input_center[:, j_idx, 1], [-1, 1, 1, 1]), 2) + tf.pow(const_z - tf.reshape(input_center[:, j_idx, 2], [-1, 1, 1, 1]), 2)) / 2.0 / stddev / stddev)
            all_vols.append(cur_vol)

        vol_labels = tf.concat(all_vols, axis=3)

    return vol_labels

if __name__ == "__main__":

    input_center = tf.placeholder(shape=[batch_size, nJoints, 3], dtype=tf.float32)

    sess = tf.Session()
    with tf.device("/device:GPU:0"):
        output_vols = build_input_3d(input_center)

    gpu_vols = sess.run(output_vols, feed_dict={input_center: np.random.random([batch_size, nJoints, 3]) * (size - 1)})

    for _ in range(evaluation_iterations):
        centers_arr = np.round(np.random.random([batch_size, nJoints, 3]) * (size - 1))
        cpu_vols = np.zeros([batch_size, size, size, size*nJoints], dtype=np.float32)

        start_time = time.clock()
        for b in range(batch_size):
            for j_idx in range(nJoints):
                cpu_vols[b, :, :, size*j_idx:size*(j_idx + 1)] = common.make_gaussian_3d(centers_arr[b, j_idx], size=64, ratio=2.0)
        start_time = time.clock() - start_time
        print(start_time)

        start_time = time.clock()
        gpu_vols = sess.run(output_vols, feed_dict={input_center: centers_arr})
        start_time = time.clock() - start_time
        print(start_time)

        # The default channel rank of tensorflow seems stupid here
        # check the gaussian_center
        for b in range(batch_size):
            for j_idx in range(nJoints):
                assert(np.argmax(cpu_vols[b, :, :, size * j_idx: size * (j_idx + 1)]) == np.argmax(gpu_vols[b, :, :, size * j_idx: size * (j_idx + 1)]))
                assert(np.argmax(cpu_vols[b, :, :, size * j_idx: size * (j_idx + 1)]) == size*size*centers_arr[b, j_idx, 1] + size*centers_arr[b, j_idx, 0] + centers_arr[b, j_idx, 2])
                # print(np.max(cpu_vols[b, :, :, size * j_idx: size * (j_idx + 1)]))
                # print(np.max(gpu_vols[b, :, :, size * j_idx: size * (j_idx + 1)]))
                assert(np.max(cpu_vols[b, :, :, size * j_idx: size * (j_idx + 1)]) == np.max(gpu_vols[b, :, :, size * j_idx: size * (j_idx + 1)]))
        # visualize the volume
        # for b in range(batch_size):
            # for j_idx in range(nJoints):
                # for i in range(size):
                    # cv2.imshow("gpu_ver", gpu_vols[b, :, :, size * j_idx: size * (j_idx + 1)][:, :, i])
                    # cv2.imshow("cpu_ver", cpu_vols[b, :, :, size * j_idx: size * (j_idx + 1)][:, :, i])
                    # cv2.waitKey()
