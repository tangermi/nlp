# -*- coding: utf-8 -*-
"""
    排序评价指标ndcg
    params
        k：
        rel_threshold：
"""

import numpy as np
import os
import math
from .base import Base
import random


class NDCG(Base):
    def __init__(self, dic_config={}, dic_engine={}, dic_score={}):
        self.dic_engine = dic_engine
        self.dic_score = dic_score
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)
        self.logger.info(self.dic_score)

    def init(self):
        self.predicted_actual_path = os.path.join(self.dic_engine['_in'], self.dic_engine['in_file'])
        self.out_file = os.path.join(self.dic_engine['_out'], self.dic_score['out_file'])
        # 这俩参数暂时不知道意思
        self.k = self.dic_score['k']
        self.rel_threshold = self.dic_score['rel_threshold']

    def load_y(self):
        self.predict_actual = np.load(self.predicted_actual_path)
        self.predict_label = self.predict_actual[0]
        self.actual_label = self.predict_actual[1]

    def _to_list(self, x):
        if isinstance(x, list):
            return x
        return [x]

    def ndcg(self):
        if self.k <= 0:
            return 0
        y_true = self._to_list(np.squeeze(self.actual_label).tolist())
        y_pred = self._to_list(np.squeeze(self.predict_label).tolist())
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        c_g = sorted(c, key=lambda x: x[0], reverse=True)
        c_p = sorted(c, key=lambda x: x[1], reverse=True)
        self.idcg = 0
        self.ndcg = 0
        for i, (g, p) in enumerate(c_g):
            if i >= self.k:
                break
            if g > self.rel_threshold:
                self.idcg += (math.pow(2, g) - 1) / math.log(2 + i)
        for i, (g, p) in enumerate(c_p):
            if i >= self.k:
                break
            if g > self.rel_threshold:
                self.ndcg += (math.pow(2, g) - 1) / math.log(2 + i)
        if self.idcg == 0:
            self.logger.info('ndcg: %s' % 0)
            return 0
        else:
            self.logger.info('ndcg: %s' % (self.ndcg / self.idcg))
            return self.ndcg / self.idcg

    def dump(self):
        pass

    def run(self):
        self.init()
        self.load_y()
        self.ndcg()
        self.dump()
