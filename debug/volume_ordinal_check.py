############### Mind this only when you are going to implement the hybird one ################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import sys
import tensorflow as tf

sys.path.append("../")

from net import ordinal_3_3

nJoints = 17
img_size = 256
feature_size = 64
flip_array = np.array([[11, 14], [12, 15], [13, 16], [1, 4], [2, 5], [3, 6]])

if __name__ == "__main__":

    input_batch_size = tf.placeholder(shape=[], dtype=tf.float32, name="batch_size")
    input_centers_2d = tf.placeholder(shape=[None, nJoints, 2], dtype=tf.float32, name="centers_2d")
    input_centers_3d = tf.placeholder(shape=[None, nJoints, 3], dtype=tf.float32, name="centers_3d")

    test_model = ordinal_3_3.mOrdinal_3_3(nJoints=nJoints, is_training=False, batch_size=input_batch_size, img_size=256)

    input_heatmaps = test_model.build_input_heatmaps(input_centers_2d, stddev=2.0, gaussian_coefficient=True)
    input_volumes = test_model.build_input_volumes(input_centers_3d, stddev=2.0, gaussian_coefficient=True)

    flip_in_flip_back_vol = test_model.flip_volumes(test_model.flip_volumes(input_volumes, flip_array), flip_array)

    reshaped_volumes = tf.transpose(tf.reshape(input_volumes, [-1, feature_size, feature_size, nJoints, feature_size]), perm=[0, 1, 2, 4, 3])
    softmaxed_volumes = tf.reshape(tf.nn.softmax(tf.reshape(reshaped_volumes, [input_batch_size, -1, nJoints]), axis=1), [input_batch_size, feature_size, feature_size, feature_size, nJoints])

    volumes_xy = tf.reduce_sum(softmaxed_volumes, axis=[3])
    volumes_z_arrs = tf.reduce_sum(softmaxed_volumes, axis=[1, 2])
    volumes_z_indices = tf.tile(np.arange(0.0, feature_size, 1.0).astype(np.float32)[np.newaxis, :, np.newaxis], [input_batch_size, 1, nJoints])

    volumes_z = tf.reduce_sum(volumes_z_arrs * volumes_z_indices, axis=1)

    with tf.Session() as sess:
        while True:
            batch_size = int(np.random.random() * 10)

            centers = np.round(np.random.random([batch_size, nJoints, 3]) * 63)

            vol_xy, vol_z, hms, vols = sess.run([volumes_xy, volumes_z, input_heatmaps, input_volumes], feed_dict={input_centers_2d: centers[:, :, 0:2], input_centers_3d: centers, input_batch_size: batch_size})

            print(np.sum(hms), np.sum(vols))
