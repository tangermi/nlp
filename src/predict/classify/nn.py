# -*- coding:utf-8 -*-
from src.predict.base import Base
import numpy as np
import tensorflow as tf


class Nn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.test_path = self.dic_engine['_in']
        self.weight_path = self.dic_engine['_train']

        self.predicted_path = self.dic_engine['_out']

    def load(self):
        with np.load(self.test_path) as train:
            self.x_test = train['x_test']
            self.y_test = train['y_test']

        with np.load(self.weight_path, allow_pickle=True) as train:
            self.weights = train['weights'].item()
            self.biases = train['biases'].item()

    def process(self):
        self.x_test = self.x_test.reshape([-1, 784])

    def neural_net(self, x):
        # Hidden fully connected layer with 128 neurons.
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        # Apply sigmoid to layer_1 output for non-linearity.
        layer_1 = tf.nn.sigmoid(layer_1)

        # Hidden fully connected layer with 256 neurons.
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Apply sigmoid to layer_2 output for non-linearity.
        layer_2 = tf.nn.sigmoid(layer_2)

        # Output fully connected layer with a neuron for each class.
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        # Apply softmax to normalize the logits to a probability distribution.
        return tf.nn.softmax(out_layer)

    def predict(self):
        self.process()
        self.res = self.neural_net(self.x_test)
        self.res = np.argmax(self.res, 1)
        # self.logger.info(self.res)
        # self.logger.info(self.y_test)

    def dump(self):
        predicted_actual = [self.res, self.y_test]
        np.save(self.predicted_path, arr=predicted_actual)

    def run(self):
        self.init()
        self.load()
        self.predict()
        self.dump()
