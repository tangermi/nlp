# -*- coding: utf-8 -*-
"""
kappa
Cohen's kappa 系数是对评分员（或标注者）间在定性（分类的）项目上的吻合性的一种统计度量
考虑到了可预见的偶然发生的吻合
"""

from sklearn.metrics import cohen_kappa_score
import os
import numpy as np
from ..base import Base


class Kappa(Base):
    def __init__(self, dic_config={}, dic_engine={}, dic_score={}):
        self.dic_engine = dic_engine
        self.dic_score = dic_score
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)
        self.logger.info(self.dic_score)

    def init(self):
        self.predicted_actual_path = self.dic_engine['_in']
        out_file = 'kappa.txt'
        if self.dic_score:
            out_file = self.dic_score.get('out_file', 'kappa.txt')
        self.out_file = os.path.join(self.dic_engine['_out'], out_file)

    def load_y(self):
        self.predict_actual = np.load(self.predicted_actual_path)
        self.predict_label = self.predict_actual[0]
        self.actual_label = self.predict_actual[1]

    def calculate_cohen_kappa(self):
        if len(np.unique(self.actual_label)) == 2:
            self.predict_label[self.predict_label >= 0.5] = 1
            self.predict_label[self.predict_label < 0.5] = 0
        self.cohen_kappa = cohen_kappa_score(self.actual_label, self.predict_label)

    def dump(self):
        with open(self.out_file, 'w', encoding='utf-8') as f:
            f.seek(0)
            f.write('cohen_kappa: %s \n' % self.cohen_kappa)
            f.truncate()

    def run(self):
        self.init()
        self.load_y()
        self.calculate_cohen_kappa()
        self.dump()
