# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np
import tensorflow as tf


# 未完成
class Nn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        # self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.hyperparams = self.dic_engine['hyperparams']
        self.epoch = self.hyperparams['epoch']

        self.weight_path = os.path.join(self.dic_engine['_out'], self.dic_engine['weight_file'])

    def load(self):
        with np.load(self.train_path) as train:
            self.x_train = train['x_train']
            self.y_train = train['y_train']

        num_features = self.hyperparams['num_features']
        batch_size = self.hyperparams['batch_size']
        self.x_train = self.x_train.reshape([-1, num_features])


        # Use tf.data API to shuffle and batch data.
        self.train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_data = self.train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

    def params(self):
        num_features = self.hyperparams['num_features']
        num_classes = self.hyperparams['num_classes']

        # Network parameters.
        n_hidden_1 = 128  # 1st layer number of neurons.
        n_hidden_2 = 256  # 2nd layer number of neurons.

        random_normal = tf.initializers.RandomNormal()

        self.weights = {
            'h1': tf.Variable(random_normal([num_features, n_hidden_1])),
            'h2': tf.Variable(random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(random_normal([n_hidden_2, num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'out': tf.Variable(tf.zeros([num_classes]))
        }

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

    # Cross-Entropy loss function.
    def cross_entropy(self, y_pred, y_true):
        num_classes = self.hyperparams['num_classes']
        # Encode label to a one hot vector.
        y_true = tf.one_hot(y_true, depth=num_classes)
        # Clip prediction values to avoid log(0) error.
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
        # Compute cross-entropy.
        return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

    # Accuracy metric.
    @staticmethod
    def accuracy(y_pred, y_true):
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    # Optimization process.
    def run_optimization(self, x, y):
        learning_rate = self.hyperparams['learning_rate']

        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            pred = self.neural_net(x)
            loss = self.cross_entropy(pred, y)

        # Variables to update, i.e. trainable variables.
        self.trainable_variables = list(self.weights.values()) + list(self.biases.values())

        # Compute gradients.
        self.gradients = g.gradient(loss, self.trainable_variables)

        # Stochastic gradient descent optimizer.
        self.optimizer = tf.optimizers.SGD(learning_rate)

        # Update W and b following gradients.
        self.optimizer.apply_gradients(zip(self.gradients, self.trainable_variables))

    def train(self):
        self.params()

        training_steps = self.hyperparams['training_steps']
        display_step = self.hyperparams['display_step']

        for step, (batch_x, batch_y) in enumerate(self.train_data.take(training_steps), 1):
            # Run the optimization to update W and b values.
            self.run_optimization(batch_x, batch_y)

            if step % display_step == 0:
                pred = self.neural_net(batch_x)
                loss = self.cross_entropy(pred, batch_y)
                acc = self.accuracy(pred, batch_y)
                self.logger.info("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

    def dump(self):
        # 保存weights和biases
        np.savez(self.weight_path, weights=self.weights, biases=self.biases)

    def run(self):
        self.init()
        self.load()
        self.train()
        self.dump()
