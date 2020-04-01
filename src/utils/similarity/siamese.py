# -*- coding:utf-8 -*-
import numpy as np


# reference from keras example "https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py"
class Similarity:
    @staticmethod
    def compute_accuracy(y_true, y_pred):
        '''
        Compute classification accuracy with a fixed threshold on distances.
        '''
        pred = y_pred.ravel() < 0.5
        return np.mean(pred == y_true)
