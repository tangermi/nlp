# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np
import pickle
import xgboost as xgb


# 训练模型
class _XGBoost(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_feature_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_feature'])
        self.train_class_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_class'])
        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])
        self.class_number = 0

    # 读取训练数据
    def load(self):
        self.train_feature = np.load(self.train_feature_path)
        self.train_class = np.load(self.train_class_path)
        self.class_number = len(set(self.train_class))

    def train(self):
        dtrain = xgb.DMatrix(self.train_feature, label=self.train_class)
        param = {'max_depth': 6, 'eta': 0.5, 'eval_metric': 'merror', 'silent': 1, 'objective': 'multi:softmax',
                 'num_class': self.class_number}  # 参数
        evallist = [(dtrain, 'train')]  # 这步可以不要，用于测试效果
        num_round = 10  # 循环次数
        self.logger.info('start training model')
        self.bst = xgb.train(param, dtrain, num_round, evallist)
        self.logger.info('model training finished')

    # save model to file
    def dump(self):
        pickle.dump(self.bst, open(self.model_path, 'wb'))

    def run(self):
        self.init()
        self.load()
        self.train()
        self.dump()
