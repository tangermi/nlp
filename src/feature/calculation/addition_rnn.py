# -*- coding:utf-8 -*-
from ..base import Base
import numpy as np
import os


class AdditionRnn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']

        self.train_feature_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_feature'])
        self.test_feature_path = os.path.join(self.dic_engine['_out'], self.dic_engine['test_feature'])

        self.hyperparams = self.dic_engine['hyperparams']
        self.chars = self.hyperparams['chars']
        self.digits = self.hyperparams['digits']

    def load(self):
        with np.load(self.data_path, allow_pickle=True) as train:
            self.x_train = train['questions']
            self.y_train = train['answers']

    # 将给定的字符串C编码成one-hot模型，参数num_rows指定这个矩阵的行数（等于问题的最大长度），列数等于字符表中总的字符数
    def encode(self, c, num_rows):
        x = np.zeros((num_rows, len(self.chars)))  # 创建一个零矩阵，行数为问题的最大长度，列数等于字符表中总的字符数
        # 针对问题字符串中的第i个字符
        for i, c in enumerate(c):
            x[i, self.char_indices[c]] = 1  # 将0矩阵的第i行和字符索引列的元素设置为1
        # 返回对字符串编码后的矩阵
        return x

    def process(self):
        self.logger.info("Vectorization...")
        MAXLEN = self.digits + 1 + self.digits

        # 初始化字符表
        self.chars = sorted(set(self.chars))
        # 创建字符表中字符和其索引的对应关系（字符：索引）
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        # 创建字符表中字符和其索引的对应关系（索引：字符）
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        # 将x初始化为（样本数，每个问题的最大长度，总的字符数）的布尔型矩阵
        x = np.zeros((len(self.x_train), MAXLEN, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(self.x_train), self.digits + 1, len(self.chars)), dtype=np.bool)
        # 针对训练问题集中的每一个问题，对其进行编码，one-hot模型，一个问题编码后对应一个矩阵
        for i, sentence in enumerate(self.x_train):
            x[i] = self.encode(sentence, MAXLEN)
        for i, sentence in enumerate(self.y_train):
            y[i] = self.encode(sentence, self.digits + 1)

        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        split_at = len(x) - len(x) // 10
        (self.x_train, self.x_val) = x[:split_at], x[split_at:]
        (self.y_train, self.y_val) = y[:split_at], y[split_at:]

    def dump(self):
        # save feature list
        np.savez(self.train_feature_path, x_train=self.x_train, y_train=self.y_train)
        np.savez(self.test_feature_path, x_test=self.x_val, y_test=self.y_val)

    def run(self):
        self.init()
        self.load()
        self.process()
        self.dump()
