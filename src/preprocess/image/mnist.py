# -*- coding: utf-8 -*-
from ..base import Base
import os
import numpy as np
from keras.utils import np_utils


'''
MNIST数据集 http://yann.lecun.com/exdb/mnist/

这里几乎没做处理，只是重新打包，把训练和测试数据分开。
In: mnist.npz
Out: train.npz, test.npz
'''
class Mnist(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']
        self.train_file = os.path.join(self.dic_engine['_out'], self.dic_engine['train_file'])
        self.test_file = os.path.join(self.dic_engine['_out'], self.dic_engine['test_file'])

    def read(self):
        with np.load(self.data_path) as mnist:
            self.x_train = mnist['x_train']
            self.y_train = mnist['y_train']
            self.x_test = mnist['x_test']
            self.y_test = mnist['y_test']

    def dump(self):
        np.savez(self.train_file, x_train=self.x_train, y_train=self.y_train)
        np.savez(self.test_file, x_test=self.x_test, y_test=self.y_test)

    def run(self):
        self.init()
        self.read()
        self.dump()



