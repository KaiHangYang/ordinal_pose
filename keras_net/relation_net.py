import numpy as np
import tensorflow as tf
import keras
import os
import sys

############### Use the tensorflow as keras backend ################
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import ReLU
from keras.layers import Softmax
from keras.optimizers import RMSprop

from keras import backend as K

sys.path.append("../")
from utils.defs.skeleton import mSkeleton15

class mRelationNet(object):

    def __init__(self, img_size, batch_size, skeleton, n_relations, name="Relation Model"):
        self.n_relations = n_relations
        self.img_size = img_size
        self.batch_size = batch_size
        self.name = name
        self.skeleton = skeleton

    def build(self):
        ############# Build the network structure first #############
        self.inputs = Input(batch_shape=(self.batch_size, self.img_size, self.img_size, 3))
        #### Use the global average pooling ####
        self.res50_model = keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=self.inputs, pooling="avg")
        net = self.res50_model.outputs[0]

        net = BatchNormalization(axis=-1)(net)
        net = ReLU()(net)
        net = Dense(units=self.n_relations * 3)(net)
        net = Reshape(target_shape=[self.n_relations, 3])(net)
        self.outputs = Softmax(axis=-1, name="outputs")(net)

        self.model = Model(inputs=self.inputs, outputs=self.outputs, name=self.name)

    def build_loss(self, learning_rate=2.5e-4, loss_type=0, alpha=[0.05, 1.0, 1.0], gamma=2.0):
        if loss_type == 0:
            # use the default categorical_crossentropy
            losses = {"outputs": "categorical_crossentropy"}
            losses_weight = {"outputs": 1.0}
        elif loss_type == 1:
            # use the focal-loss-like loss functions
            def focal_loss(y_true, y_pred):
                ## The y_pred is the value after softmax
                const_alpha = K.constant(alpha)
                const_alpha = K.tile(const_alpha[tf.newaxis, tf.newaxis], [self.batch_size, self.n_relations, 1])
                return K.sum((-1.0 * K.pow(1.0 - y_pred, float(gamma))) * K.log(y_pred + K.epsilon()) * y_true * const_alpha, axis=-1)

            losses = {"outputs": focal_loss}
            losses_weight = {"outputs": 1.0}
        else:
            print("Invalid loss type!")
            quit()

        #### The metrics function ####
        def mean_accuracy(y_true, y_pred):
            return K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), tf.float32))
        self.model.compile(optimizer=RMSprop(lr=learning_rate), loss=losses, loss_weights=losses_weight, metrics=[mean_accuracy])

    def set_lr(self, learning_rate=2.5e-4):
        K.set_value(self.model.optimizer.lr, learning_rate)

    def get_lr(self):
        return K.get_value(self.model.optimizer.lr)

    def train_on_batch(self, x, y):
        loss, accuracy = self.model.train_on_batch(x, y)

        return loss, accuracy
    def test_on_batch(self, x, y):
        loss, accuracy = self.model.test_on_batch(x, y)
        return loss, accuracy

    def save_model(self, path):
        self.model.save_weights(path)

    def restore_model(self, path):
        self.model.load_weights(path)
