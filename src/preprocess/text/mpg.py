# -*- coding:utf-8 -*-
from ..base import Base
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np


'''
# mpg数据集 https://archive.ics.uci.edu/ml/datasets/auto+mpg

这里对数据进行了get dummies处理， 把产地变成了美国/日本/欧洲。 同时也对数据进行了归一化处理。

In: [1, 2, 3, 43, 45, 45, 34, 1]
Out: [0.3, 0.45. 0.3. 0.2, 0.6, 0,23, 0,23, 0.345, 0, 0]

'''
# 回归
class Mpg(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.dataset_path = self.dic_engine['_in']
        self.train_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_out'], self.dic_engine['test_file'])
        self.mean_std_path = os.path.join(self.dic_engine['_out'], self.dic_engine['mean_std_file'])
        # 拼接输出路径, 输出图片
        self.img_path = os.path.join(self.dic_engine['_out'], self.dic_engine['out_img'])
        self.dataset = None
        self.normed_train_data = None
        self.normed_test_data = None
        self.train_labels = None
        self.test_labels = None

    def load(self):
        dataset_path = self.dataset_path

        # 使用Pandas读取数据
        column_names = ['MPG', '气缸', '排量', '马力', '重量', '加速度', '年份', '产地']
        raw_dataset = pd.read_csv(dataset_path, names=column_names,
                                  na_values="?", comment='\t',
                                  sep=" ", skipinitialspace=True)

        self.dataset = raw_dataset.copy()

    def process(self):
        dataset = self.dataset
        dataset = dataset.dropna()

        dataset = pd.get_dummies(dataset, columns=['产地'])
        dataset = dataset.rename(columns={'产地_1': '美国', '产地_2': '欧洲', '产地_3': '日本'})
        # 看一看转换后的结果
        # self.logger.info(dataset.head(3))

        # 训练集 80%， 测试集 20%
        train_dataset = dataset.sample(frac=0.8, random_state=0)
        self.train_dataset = train_dataset.copy()
        test_dataset = dataset.drop(train_dataset.index)

        train_stats = train_dataset.describe()
        train_stats.pop("MPG")
        train_stats = train_stats.transpose()
        self.mean = train_stats['mean'].to_numpy()
        self.std = train_stats['std'].to_numpy()

        # 分离 label
        self.train_labels = train_dataset.pop('MPG').to_numpy()
        self.test_labels = test_dataset.pop('MPG').to_numpy()

        def norm(x):
            return (x - train_stats['mean']) / train_stats['std']

        self.normed_train_data = norm(train_dataset).to_numpy()
        self.normed_test_data = norm(test_dataset).to_numpy()

    def plot(self):
        # 解决中文乱码问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 看一看训练集中属性两两之间的关系
        sns_plot = sns.pairplot(self.train_dataset[["MPG", "气缸", "排量", "重量"]], diag_kind="kde")
        sns_plot.savefig(self.img_path)

    def dump(self):
        np.savez(self.train_path,
                 normed_train_data=self.normed_train_data,
                 train_labels=self.train_labels)

        np.savez(self.test_path,
                 normed_test_data=self.normed_test_data,
                 test_labels=self.test_labels)

        np.savez(self.mean_std_path,
                 mean=self.mean,
                 std=self.std)

    def run(self):
        self.init()
        self.load()
        self.process()
        self.plot()
        self.dump()
