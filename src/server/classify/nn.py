# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image


class Nn:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.weight_path = dic_config['weight_path']

    def load(self):
        with np.load(self.weight_path, allow_pickle=True) as train:
            self.weights = train['weights'].item()
            self.biases = train['biases'].item()

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

    def process(self, img):
        self.img = np.array([np.array(Image.open(img).convert('L').resize((28, 28))).astype('float32') / 255.0])
        self.img = self.img.reshape([-1, 784])

    def _predict(self, img):
        self.process(img)
        self.res = self.neural_net(self.img)
        self.res = np.argmax(self.res, 1)
        return self.res
