# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np
import random


'''
# 输入fashion.mnist数据集
# 输出相对分类和不同分类的图片对

In: [image1, image2, image3...]  [label1, label2, label3...]
Out: [[[image1, image2], [image1, image102]], [[image2, image3], [image2, image302]]...] 
     [[0, 1], [0, 1]...]
     假设image 1-100 的类别为'猫'， image 101-200 的类别为'狗' [[猫，猫], [猫，狗]] 对应一个 [[0, 1]]
'''
class Siamese(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']
        # 拼接保存路径
        self.train_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_out'], self.dic_engine['test_file'])

    # 处理一个txt
    def load(self):
        with np.load(self.data_path) as data:
            self.x_train = data['x_train']
            self.y_train = data['y_train']

    def create_pairs(self, x, digit_indices):
        '''
        Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        # labels are 1 or 0 identify whether the pair is positive or negative
        labels = []

        class_num = digit_indices.shape[0]
        for d in range(class_num):
            for i in range(int(digit_indices.shape[1]) - 1):
                # use images from the same class to create positive pairs
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                # use random number to find images from another class to create negative pairs
                inc = random.randrange(1, class_num)
                dn = (d + inc) % class_num
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                # add two labels which the first one is positive class and the second is negative.
                labels += [1, 0]
        return np.array(pairs), np.array(labels)

    def process(self):
        # 把数据根据标签切分为不同的组，用作测试
        # ["top", "trouser", "pullover", "coat", "sandal", "ankle boot"]的位置在 [0,1,2,4,5,9]
        # split labels ["top", "trouser", "pullover", "coat", "sandal", "ankle boot"] to train set
        digit_indices = [np.where(self.y_train == i)[0] for i in {0, 1, 2, 4, 5, 9}]
        digit_indices = np.array(digit_indices)

        # 每一个分类的内容条数
        n = min([len(digit_indices[d]) for d in range(6)])
        # ["dress", "sneaker", "bag", "shirt"]的位置在 [3,6,7,8]
        # 使用80%的标签为 ["top", "trouser", "pullover", "coat", "sandal", "ankleboot"] 用作训练
        train_set_shape = n * 0.8
        y_train_new = digit_indices[:, :int(train_set_shape)]
        y_test_new = digit_indices[:, int(train_set_shape):]

        # For training, images are from 80% of the images with labels ["top", "trouser", "pullover", "coat",
        # "sandal", "ankleboot"]
        tr_pairs, tr_y = self.create_pairs(self.x_train, y_train_new)
        # Reshape for the convolutional neural network, same for the test sets below.
        tr_pairs = tr_pairs.reshape(tr_pairs.shape[0], 2, 28, 28, 1)
        # self.logger.info(tr_pairs.shape)

        # For testing, images are from the rest 20% of the images with labels ["top", "trouser", "pullover", "coat",
        # "sandal", "ankleboot"]
        te_pairs_1, te_y_1 = self.create_pairs(self.x_train, y_test_new)
        te_pairs_1 = te_pairs_1.reshape(te_pairs_1.shape[0], 2, 28, 28, 1)
        # self.logger.info(te_pairs_1.shape)

        # 使用100%的标签为["dress", "sneaker", "bag", "shirt"]的数据作用测试
        digit_indices_t = [np.where(self.y_train == i)[0] for i in {3, 6, 7, 8}]
        y_test_new_2 = np.array(digit_indices_t)
        te_pairs_2, te_y_2 = self.create_pairs(self.x_train, y_test_new_2)
        te_pairs_2 = te_pairs_2.reshape(te_pairs_2.shape[0], 2, 28, 28, 1)
        # self.logger.info(te_pairs_2.shape)

        # Keep 100% of the images with labels ["top", "trouser", "pullover", "coat","sandal", "ankle boot"] union [
        # "dress", "sneaker", "bag", "shirt"] for testing
        digit_indices_3 = [np.where(self.y_train == i)[0] for i in range(10)]
        y_test_new_3 = np.array(digit_indices_3)
        te_pairs_3, te_y_3 = self.create_pairs(self.x_train, y_test_new_3)
        te_pairs_3 = te_pairs_3.reshape(te_pairs_3.shape[0], 2, 28, 28, 1)
        # self.logger.info(te_pairs_3.shape)

        self.train = [tr_pairs, tr_y]
        self.test = [[te_pairs_1, te_y_1], [te_pairs_2, te_y_2], [te_pairs_3, te_y_3]]

    def dump(self):
        np.savez(self.train_path, tr_pairs=self.train[0], tr_y=self.train[1])
        np.savez(self.test_path, te_pairs_1=self.test[0][0], te_y_1=self.test[0][1],
                 te_pairs_2=self.test[1][0], te_y_2=self.test[1][1],
                 te_pairs_3=self.test[2][0], te_y_3=self.test[2][1], )

    def run(self):
        self.init()
        self.load()
        self.process()
        self.dump()
