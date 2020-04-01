# -*- coding: utf-8 -*-
"""
Jaccard
两个集合的交集的大小除以两个集合的并集的大小
"""

import os
import numpy as np
from ..base import Base
from sklearn.metrics import jaccard_similarity_score


class Jaccard(Base):
    def __init__(self, dic_config={}, dic_engine={}, dic_score={}):
        self.dic_engine = dic_engine
        self.dic_score = dic_score
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)
        self.logger.info(self.dic_score)

    def init(self):
        self.predicted_actual_path = self.dic_engine['_in']
        out_file = 'jaccard.txt'
        if self.dic_score:
            out_file = self.dic_score.get('out_file', 'jaccard.txt')
        self.out_file = os.path.join(self.dic_engine['_out'], out_file)

    def load_y(self):
        self.predict_actual = np.load(self.predicted_actual_path)
        self.predict_label = self.predict_actual[0]
        self.actual_label = self.predict_actual[1]

    def calculate_jaccard(self):
        if len(np.unique(self.actual_label)) == 2:
            self.predict_label[self.predict_label >= 0.5] = 1
            self.predict_label[self.predict_label < 0.5] = 0
        self.Jaccard_index = jaccard_similarity_score(self.actual_label, self.predict_label)

    def dump(self):
        with open(self.out_file, 'w', encoding='utf-8') as f:
            f.seek(0)
            f.write('Jaccard similarity: %s \n' % self.Jaccard_index)
            f.truncate()

    def run(self):
        self.init()
        self.load_y()
        self.calculate_jaccard()
        self.dump()
