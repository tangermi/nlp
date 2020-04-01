# -*- coding: utf-8 -*-
from ..base import Base
import os
import numpy as np
import matplotlib.pyplot as plt

'''
# fashion MINST数据集 https://www.kaggle.com/zalando-research/fashionmnist

In: 目录路径   fashion_mnist.npz
    [image1, image2, image3...]   max(image) = 255, min(image) = 0
Out: 训练集和测试集   train.npz, test.npz
    [image1, image2, image3...]   max(image) = 1, min(image) = 0
'''
class FashionMnist(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']
        self.train_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_out'], self.dic_engine['test_file'])

        self.train_overview_path = ''
        if self.dic_engine.get('train_overview_file'):
            self.train_overview_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_overview_file'])

    def read(self):
        with np.load(self.data_path) as fashion_mnist:
            self.x_train = fashion_mnist['train_images']
            self.y_train = fashion_mnist['train_labels']

            self.x_test = fashion_mnist['test_images']
            self.y_test = fashion_mnist['test_labels']

    def plot_overview(self):
        # 图片概览
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.x_train[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[self.y_train[i]])
        plt.savefig(self.train_overview_path)

    def process(self):
        # 归一化处理
        self.x_train = self.x_train.astype('float32')
        self.x_train = self.x_train / 255.0

        self.x_test = self.x_test.astype('float32')
        self.x_test = self.x_test / 255.0

        if self.train_overview_path:
            self.plot_overview()

    def dump(self):
        np.savez(self.train_path, x_train=self.x_train, y_train=self.y_train)
        np.savez(self.test_path, x_test=self.x_test, y_test=self.y_test)

    def run(self):
        self.init()
        self.read()
        self.process()
        self.dump()
