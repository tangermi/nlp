# -*- coding:utf-8 -*-
from src.predict.base import Base
import pickle
import xgboost as xgb
import numpy as np


class _XGBoost(Base):
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
        self.bst = pickle.load(open(self.model_path, 'rb'))
        with np.load(self.data_path) as test:
            self.test_feature = test['test_feature']
            self.test_class = test['test_class']

    def predict(self):
        dtest = xgb.DMatrix(self.test_feature)
        res = self.bst.predict(dtest)
        self.predict_actual = [res, self.test_class]

    def dump(self):
        np.save(self.predicted_path, arr=self.predict_actual)

    def run(self):
        self.init()
        self.load()
        self.predict()
        self.dump()
