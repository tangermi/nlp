# -*- coding: utf-8 -*-
from ..base import Base
import os
import numpy as np
import json

'''
# imdb数据集 https://keras.rstudio.com/reference/dataset_imdb.html

这里没有做处理，只是将训练数据和测试数据读取后分开打包，附带有tensorflow提供的index
out: train.npz, test.npz, index.npy
'''
class Imdb(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']
        self.index_in = self.dic_engine['index_in']

        self.train_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_out'], self.dic_engine['test_file'])
        self.index_path = os.path.join(self.dic_engine['_out'], self.dic_engine['index_file'])

    def read(self):
        with np.load(self.data_path, allow_pickle=True) as imdb:
            self.x_train = imdb['x_train']
            self.y_train = imdb['y_train']

            self.x_test = imdb['x_test']
            self.y_test = imdb['y_test']

        self.word_index = json.load(open(self.index_in))

    def process(self):
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

    def dump(self):
        np.savez(self.train_path, x_train=self.x_train, y_train=self.y_train)
        np.savez(self.test_path, x_test=self.x_test, y_test=self.y_test)
        np.save(self.index_path, arr=self.word_index)

    def run(self):
        self.init()
        self.read()
        self.process()
        self.dump()
