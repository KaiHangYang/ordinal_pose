import numpy as np
import tensorflow as tf
import os
import sys

from network_utils import mResidualUtils

def build_hourglass(inputs, nOut=256, nPooling=4, name='hourglass', is_training=True):
    res_utils = mResidualUtils(is_training=is_training)
    with tf.variable_scope(name):
        # encoding block
        cur_input = inputs
        encoding_blocks_out = []
        skip_blocks_out = []

        for i in range(nPooling):
            with tf.variable_scope("encoding_{}".format(i+1)):
                for j in range(3):
                    cur_input = res_utils.residual_block(cur_input, nOut, kernel_size=3, strides=1, name="res{}".format(j+1))

                encoding_blocks_out.append(cur_input)
                cur_input = tf.layers.max_pooling2d(cur_input, pool_size=2, strides=2, padding="VALID", name="down_sampling")

        # skip block
        for i in range(nPooling):
            with tf.variable_scope("skip_{}".format(i+1)):
                res = res_utils.residual_block(encoding_blocks_out[i], nOut, kernel_size=3, strides=1, name="res1")
                res = res_utils.residual_block(res, nOut, kernel_size=3, strides=1, name="res2")
                res = res_utils.residual_block(res, nOut, kernel_size=3, strides=1, name="res3")

                skip_blocks_out.append(res)

        # inner main path
        with tf.variable_scope("encoding_{}".format(nPooling+1)):
            for j in range(3):
                cur_input = res_utils.residual_block(cur_input, nOut, kernel_size=3, strides=1, name="res{}".format(j+1))

        with tf.variable_scope("decoding_{}".format(1)):
            cur_input = res_utils.residual_block(cur_input, nOut, kernel_size=3, strides=1, name="res1")

        skip_blocks_out = skip_blocks_out[::-1]
        # decoding block
        for i in range(nPooling):
            with tf.variable_scope("decoding_{}".format(i+2)):
                with tf.variable_scope("upsampling"):
                    cur_shape = cur_input.get_shape()[1:3].as_list()
                    cur_input = tf.image.resize_nearest_neighbor(cur_input, [cur_shape[0] * 2, cur_shape[1] * 2], name="up_sampling")
                cur_input = tf.add_n([cur_input, skip_blocks_out[i]], name="elw_add")

                cur_input = res_utils.residual_block(cur_input, nOut, kernel_size=3, strides=1, name="res1")

        return cur_input

if __name__ == "__main__":
    with tf.Session() as sess:
        input = tf.placeholder(shape=[1, 256, 256, 3], dtype=tf.float32)
        hg = build_hourglass(input)

        summary_writer = tf.summary.FileWriter(logdir="./log/", graph=sess.graph)

        summary_writer.close()
