# -*- coding:utf-8 -*-
from ..base import Base
import numpy as np
import os


'''
# 输入mnist手写数字
# 输出rescale到[-1, 1]后的数据

In: [image1, image2, image3...]   max(image) = 255, min(image) = 0
Out: [image1, image2, image3...]   max(image) = 1, min(image) = -1
'''
class Gan(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.train_feature_path = os.path.join(self.dic_engine['out'], self.dic_engine['train_feature'])
        self.test_feature_path = os.path.join(self.dic_engine['out'], self.dic_engine['test_feature'])

    def load(self):
        with np.load(self.train_path, allow_pickle=True) as train:
            self.x_train = train['x_train']
            self.y_train = train['y_train']

        with np.load(self.test_path, allow_pickle=True) as test:
            self.x_test = test['x_test']
            self.y_test = test['y_test']

    def process(self):
        self.x_train = (self.x_train.astype(np.float32) - 127.5) / 127.5
        self.x_train = np.expand_dims(self.x_train, axis=-1)

        self.x_test = (self.x_test.astype(np.float32) - 127.5) / 127.5
        self.x_test = np.expand_dims(self.x_test, axis=-1)

    def dump(self):
        # save feature list
        np.savez(self.train_feature_path, x_train=self.x_train, y_train=self.y_train)
        np.savez(self.test_feature_path, x_test=self.x_test, y_test=self.y_test)

    def run(self):
        self.init()
        self.load()
        self.process()
        self.dump()
