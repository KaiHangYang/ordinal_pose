import numpy as np
import tensorflow as tf
import os
import sys

from network_utils import mResidualUtils

def build_hourglass(inputs, nOut=256, nPooling=4, name='hourglass', is_training=True, res_utils=None):
    if res_utils is None:
        print("Use the default resblock settings!")
        res_utils = mResidualUtils(is_training=is_training, is_tiny=False, is_use_bias=True)

    with tf.variable_scope(name):
        # encoding block
        up1 = inputs

        with tf.variable_scope("up_1"):
            for i in range(3):
                up1 = res_utils.residual_block(up1, nOut, name="res{}".format(i))

        with tf.variable_scope("low_1"):
            low1 = tf.layers.max_pooling2d(inputs, pool_size=2, strides=2, padding="VALID", name="down_samplint")
            for i in range(3):
                low1 = res_utils.residual_block(low1, nOut, name="res{}".format(i))

        if nPooling > 1:
            low2 = build_hourglass(low1, nOut, nPooling-1, name="inner_hg")
        else:
            low2 = low1
            with tf.variable_scope("mid"):
                for i in range(3):
                    low2 = res_utils.residual_block(low2, nOut, name="res{}".format(i))

        with tf.variable_scope("low_2"):
            low3 = res_utils.residual_block(low2, nOut, name="res")

        with tf.variable_scope("up_2"):
            cur_shape = low3.get_shape()[1:3].as_list()
            up2 = tf.image.resize_nearest_neighbor(low3, [cur_shape[0] * 2, cur_shape[1] * 2], name="up_sampling")

        return tf.add_n([up1, up2], name="out_hg")

if __name__ == "__main__":
    with tf.Session() as sess:
        input = tf.placeholder(shape=[1, 64, 64, 3], dtype=tf.float32)
        hg = build_hourglass(input)

        summary_writer = tf.summary.FileWriter(logdir="./log/", graph=sess.graph)

        summary_writer.close()
