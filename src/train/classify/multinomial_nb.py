# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pickle


# 训练模型
class _MultinomialNB(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_feature_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_feature'])
        self.train_class_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_class'])

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])

    # 读取训练数据
    def load(self):
        self.nb = MultinomialNB()
        self.train_feature_list = np.load(self.train_feature_path)
        self.train_class_list = np.load(self.train_class_path)

    def train(self):
        self.classifier = self.nb.fit(self.train_feature_list, self.train_class_list)

    def dump(self):
        # save model to file
        pickle.dump(self.classifier, open(self.model_path, 'wb'))

    def run(self):
        self.init()
        self.load()
        self.train()
        self.dump()

