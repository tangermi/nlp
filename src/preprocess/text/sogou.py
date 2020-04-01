# -*- coding:utf-8 -*-
from ..base import Base
import os
import pandas as pd
import random


'''
# sogou新闻mini数据集 http://www.sogou.com/labs/resource/cs.php
读取文件夹树的txt文件
输出为csv格式的文件

In: 我要吃饭了
Out: ['我', '要', '吃饭', '了']
'''
# 数据预处理
class Sogou(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_dir = self.dic_engine['_in']  # 训练集地址
        self.train_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_out'], self.dic_engine['test_file'])

    # 处理一个txt
    def read_txt(self, new_data_dir, file, folder):
        with open(os.path.join(new_data_dir, file), 'rb+') as f:
            raw = f.read()
            raw = raw.decode('utf8')

            self.data_list.append(raw)  # 添加数据集数据
            self.class_list.append(folder)  # 添加数据集类别

    def read(self):
        folder_list = os.listdir(self.data_dir)  # 查看data_dir下的文件
        self.data_list = []  # 数据集数据
        self.class_list = []  # 数据集类别

        # 遍历每个子文件夹
        for folder in folder_list:
            new_data_dir = os.path.join(self.data_dir, folder)  # 根据子文件夹，生成新的路径
            if os.path.isdir(new_data_dir):
                self.logger.info('load folder: ' + str(folder))
                files = os.listdir(new_data_dir)  # 存放子文件夹下的txt文件列表
            else:
                continue
            j = 1
            # 遍历每个txt文件
            for file in files:
                if j > 400:
                    break
                self.read_txt(new_data_dir, file, folder)
                j += 1

    # 乱序并切分为训练集与测试集
    def process(self):
        data_class_list = list(zip(self.data_list, self.class_list))  # zip压缩合并，将数据与标签对应压缩
        random.shuffle(data_class_list)  # 将data_class_list乱序
        test_size = 0.2
        index = int(len(data_class_list) * test_size) + 1  # 训练集和测试集切分的索引值
        train_list = data_class_list[index:]  # 训练集
        test_list = data_class_list[:index]  # 测试集
        self.train_df = pd.DataFrame(train_list, columns=['text', 'class'])
        self.test_df = pd.DataFrame(test_list, columns=['text', 'class'])

        # self.logger.info(train_df.head())
        # self.logger.info(train_df.info())

    def dump(self):
        # 输出为csv格式的文件
        self.train_df.to_csv(self.train_path, sep=',', encoding='utf-8')
        self.test_df.to_csv(self.test_path, sep=',', encoding='utf-8')

    def run(self):
        self.init()
        self.read()
        self.process()
        self.dump()
