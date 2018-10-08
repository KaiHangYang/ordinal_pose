import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
import sys
import time

sys.path.append("../")
from utils.preprocess_utils import common
from utils.postprocess_utils import volume_utils

# volume shape (batch_size, feature_size, feature_size, feature_size * nJoints)

nJoints = 17
feature_size = 64
batch_size = 4

# batch_size 4 100ms
# def get_joints_3d(volumes, batch_size, name="volume_to_joints"):
    # # volume shape (batch_size, feature_size, feature_size, feature_size * nJoints)
    # with tf.device("/device:GPU:0"):
        # with tf.variable_scope(name):
            # all_joints = []
            # for i in range(nJoints):
                # cur_volume = volumes[:, :, :, feature_size*i:feature_size*(i+1)]
                # cur_argmax_index = tf.argmax(tf.layers.flatten(cur_volume), axis=1)

                # with tf.device("cpu:0"):
                    # cur_joints = tf.transpose(tf.unravel_index(cur_argmax_index, [feature_size, feature_size, feature_size]))[:, tf.newaxis]
                # all_joints.append(tf.concat([cur_joints[:, :, 0:2][:, :, ::-1], cur_joints[:, :, 2][:, :, tf.newaxis]], axis=2))

            # return tf.cast(tf.concat(all_joints, axis=1), tf.float32)

# batch_size 4 40ms
def get_joints_3d(volumes, batch_size, name="volume_to_joints"):
    # volume shape (batch_size, feature_size, feature_size, feature_size * nJoints)
    with tf.device("/device:GPU:0"):
        with tf.variable_scope(name):
            cur_volumes = tf.reshape(tf.transpose(tf.reshape(volumes, [batch_size, feature_size, feature_size, nJoints, feature_size]), perm=[0, 1, 2, 4, 3]), [batch_size, -1, nJoints])
            cur_argmax_index = tf.reshape(tf.argmax(cur_volumes, axis=1), [-1])

            with tf.device("/cpu:0"):
                cur_joints = tf.unravel_index(cur_argmax_index, [feature_size, feature_size, feature_size])

            cur_joints = tf.reshape(tf.transpose(cur_joints), [-1, nJoints, 3])
            cur_joints = tf.concat([cur_joints[:, :, 0:2][:, :, ::-1], cur_joints[:, :, 2][:, :, tf.newaxis]], axis=2)
            return tf.cast(cur_joints, tf.float32)

if __name__ == "__main__":

    input_volume = tf.placeholder(shape=[batch_size, 64, 64, 64*17], dtype=tf.float32)
    output_joints = get_joints_3d(input_volume, batch_size =batch_size)

    with tf.Session() as sess:
        while True:
            batch_volume_np = np.zeros([batch_size, 64, 64, 64 * 17], dtype=np.float32)
            batch_joints_np = (np.random.random([batch_size, 17, 3]) * 63).astype(np.int32)

            for b in range(batch_size):
                for j in range(17):
                    batch_volume_np[b, :, :, 64*j:64*(j+1)] = common.make_gaussian_3d(batch_joints_np[b, j])

            get_time = time.clock()
            get_3d = sess.run(output_joints, feed_dict={input_volume: batch_volume_np})
            get_time = time.clock() - get_time

            assert((get_3d == batch_joints_np).all())
            print(get_time)
