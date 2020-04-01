# -*- coding:utf-8 -*-
from src.predict.base import Base
import pickle
import os
import numpy as np


class _MultinomialNB(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']
        # self.test_class_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_class'])
        self.model_path = self.dic_engine['_train']
        self.predicted_path = self.dic_engine['_out']

    def load(self):
        self.nb = pickle.load(open(self.model_path, 'rb'))
        with np.load(self.data_path) as test:
            self.test_feature = test['test_feature_list']
            self.test_calss = test['test_class_list']

    def predict(self):
        res = self.nb.predict(self.test_feature)
        self.predicted_actual = [res, self.test_calss]

    def dump(self):
        np.save(self.predicted_path, arr=self.predicted_actual)

    def run(self):
        self.init()
        self.load()
        self.predict()
        self.dump()
