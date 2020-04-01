# -*- coding: utf-8 -*-
from ..base import Base
import random
import os
import numpy as np
import matplotlib.pyplot as plt


'''
Cifar10数据集 https://www.cs.toronto.edu/~kriz/cifar.html
数据集含有60000张32x32图片，分为10个分类，每个分类包含6000张图片。 50000张被用来做训练，10000张用来测试

In:  目录路径   cifar-10-batches-py/
     [image1, image2, image3...]   max(image) = 255, min(image) = 0
Out: 训练集和测试集   train.npz, test.npz 
    [image1, image2, image3...]   max(image) = 1, min(image) = 0
'''
class Cifar10(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        # 训练数据
        self.data_path = self.dic_engine['_in']
        self.train_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_file'])

        # 测试数据
        self.test_data_path = ''
        self.test_path = ''
        if self.dic_engine.get('_test_in'):
            self.test_data_path = self.dic_engine.get('_test_in')
            self.test_path = os.path.join(self.dic_engine['_out'], self.dic_engine['test_file'])

        # 所有类的综合概览
        self.train_overview_path = ''
        if self.dic_engine.get('train_overview_file'):
            self.train_overview_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_overview_file'])

    @staticmethod
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def preocess_data(self):
        batch_1 = self.unpickle(self.data_path)
        self.train_images = batch_1[b'data']
        self.train_labels = batch_1[b'labels']
        # self.logger.info(self.train_images.shape)

        self.train_images = self.train_images.reshape((len(self.train_images), 3, 32, 32)).transpose(0, 2, 3, 1)
        if self.train_overview_path:
            self.plot_overview()
        self.train_images = self.train_images / 255.0

    def preocess_test_data(self):
        test = self.unpickle(self.test_data_path)
        self.test_images = test[b'data']
        self.test_labels = test[b'labels']
        # self.logger.info(len(self.train_labels))

        self.test_images = self.test_images.reshape((len(self.test_images), 3, 32, 32)).transpose(0, 2, 3, 1)
        self.test_images = self.test_images / 255.0

    def plot_overview(self):
        # 通过plot前25条数据来验证数据
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            # The CIFAR labels happen to be arrays,
            # which is why you need the extra index
            plt.xlabel(class_names[self.train_labels[i]])
        plt.savefig(self.train_overview_path)
        plt.show()

    def process(self):
        self.preocess_data()
        if self.test_data_path:
            self.preocess_test_data()

        # self.train_images = self.normalize(self.train_images)
        # self.train_labels = self.one_hot_encode(self.train_labels)

    def dump(self):
        np.savez(self.train_path, train_images=self.train_images, train_labels=self.train_labels)
        if self.test_path:
            np.savez(self.test_path, test_images=self.test_images, test_labels=self.test_labels)

    def run(self):
        self.init()
        self.process()
        self.dump()
