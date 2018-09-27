import numpy as np
import os
import sys
import tensorflow as tf

class mResidualUtils(object):
    def __init__(self, is_training=True, is_tiny=False, is_use_bias=True):
        self.is_training = is_training
        self.is_tiny = is_tiny
        self.is_use_bias = is_use_bias

    def _main_path(self, inputs, nOut, kernel_size=3, strides=1, name='main_path'):
        if self.is_tiny:
            with tf.variable_scope(name):
                norm = tf.contrib.layers.batch_norm(inputs, 0.9, center=True, scale=True, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.is_training)
                conv = tf.layers.conv2d(norm, nOut, kernel_size, strides, padding="SAME", use_bias=self.is_use_bias, name="conv")
                return conv
        else:
            with tf.variable_scope(name):
                with tf.variable_scope("norm_1"):
                    norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, center=True, scale=True, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.is_training)
                    conv_1 = tf.layers.conv2d(norm_1, nOut/2, 1, 1, padding="SAME", use_bias=self.is_use_bias, name="conv")
                with tf.variable_scope("norm_2"):
                    norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, center=True, scale=True, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.is_training)
                    conv_2 = tf.layers.conv2d(norm_2, nOut/2, kernel_size, strides, padding="SAME", use_bias=self.is_use_bias, name="conv")
                with tf.variable_scope("norm_3"):
                    norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, center=True, scale=True, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.is_training)
                    conv_3 = tf.layers.conv2d(norm_3, nOut, 1, 1, padding="SAME", use_bias=self.is_use_bias, name="conv")

                return conv_3

    def _skip_path(self, inputs, nOut, strides=1, name='skip_path'):
        with tf.variable_scope(name):
            if inputs.get_shape().as_list()[3] == nOut and strides == 1:
                return inputs # the num of input and output matches
            else:
                conv = tf.layers.conv2d(inputs, nOut, 1, strides, padding="SAME", use_bias=self.is_use_bias, name="conv")
                return conv

    def residual_block(self, inputs, nOut, kernel_size=3, strides=1, name='residual_block'):
        with tf.variable_scope(name):
            main_path = self._main_path(inputs, nOut, kernel_size, strides)
            skip_path = self._skip_path(inputs, nOut, strides)

            return tf.add_n([main_path, skip_path], name="elw_add")
