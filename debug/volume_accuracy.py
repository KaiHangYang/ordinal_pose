import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import tensorflow as tf
import sys

sys.path.append("../")
from utils.preprocess_utils import common
from utils.postprocess_utils import volume_utils

# volume shape (batch_size, feature_size, feature_size, feature_size * nJoints)
def get_joints(volumes, nJoints=17):
    all_joints = []
    for i in range(nJoints):
        cur_volume = volumes[:, :, :, 64*i:64*(i+1)]
        cur_joints = tf.transpose(tf.unravel_index(tf.argmax(tf.layers.flatten(cur_volume), axis=1), [64, 64, 64]))[:, np.newaxis]
        all_joints.append(tf.concat([cur_joints[:, :, 0:2][:, :, ::-1], cur_joints[:, :, 2][:, :, np.newaxis]], axis=2))
    return tf.concat(all_joints, axis=1)

if __name__ == "__main__":

    input_volume = tf.placeholder(shape=[4, 64, 64, 64*17], dtype=tf.float32)
    output_joints = get_joints(input_volume, nJoints=17)

    with tf.Session() as sess:
        while True:
            batch_volume_np = np.zeros([4, 64, 64, 64 * 17], dtype=np.float32)
            batch_joints_np = (np.random.random([4, 17, 3]) * 63).astype(np.int32)

            for b in range(4):
                for j in range(17):
                    batch_volume_np[b, :, :, 64*j:64*(j+1)] = common.make_gaussian_3d(batch_joints_np[b, j])

            a = sess.run(output_joints, feed_dict={input_volume: batch_volume_np})

            print(np.max(np.abs(a - batch_joints_np)))
