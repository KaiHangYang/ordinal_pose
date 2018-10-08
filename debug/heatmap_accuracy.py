import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
import sys

sys.path.append("../")
from utils.preprocess_utils import common
# heatmaps shape (batch_size, feature_size, feature_size, feature_size * nJoints)

# heatmaps shape (batch_size, 64, 64, 17)
def get_joints_2d(heatmaps, nJoints=17, batch_size=4, name="get_joints_hm"):
    with tf.variable_scope(name):
        max_indices = tf.argmax(tf.reshape(heatmaps, [batch_size, -1, 17]), axis=1)
        cur_joints = tf.reshape(tf.transpose(tf.unravel_index(tf.reshape(max_indices, [-1]), [64, 64])), [-1, 17, 2])[:, :, ::-1]
    return cur_joints

if __name__ == "__main__":

    input_heatmaps = tf.placeholder(shape=[4, 64, 64, 17], dtype=tf.float32)
    output_joints = get_joints_2d(input_heatmaps, nJoints=17)

    with tf.Session() as sess:
        while True:
            batch_heatmaps_np = np.zeros([4, 64, 64, 17], dtype=np.float32)
            batch_joints_np = (np.random.random([4, 17, 2]) * 63).astype(np.int32)

            for b in range(4):
                for j in range(17):
                    batch_heatmaps_np[b, :, :, j] = common.make_gaussian(batch_joints_np[b, j])

            get_2d = sess.run(output_joints, feed_dict={input_heatmaps: batch_heatmaps_np})
            assert((get_2d == batch_joints_np).all())
