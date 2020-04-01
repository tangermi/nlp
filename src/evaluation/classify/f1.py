# -*- coding: utf-8 -*-
import pickle
from sklearn import metrics
from ..base import Base
import os
import numpy as np


class F1(Base):
    """f1, precision_score, recall_score
            params: actual_label :1d array or sparse matrix
                    predict_label : 1d array or sparse matrix
    """

    def __init__(self, dic_config={}, dic_engine={}, dic_score={}):
        self.dic_engine = dic_engine
        self.dic_score = dic_score
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)
        self.logger.info(self.dic_score)

    def init(self):
        self.average = self.dic_score['average']
        self.predicted_actual_path = self.dic_engine['_in']
        out_file = 'f1_score.txt'
        if self.dic_score:
            out_file = self.dic_score.get('out_file', 'f1_score.txt')
        self.out_file = os.path.join(self.dic_engine['_out'], out_file)

    def load_y(self):
        self.predict_actual = np.load(self.predicted_actual_path)
        self.predict_label = self.predict_actual[0]
        self.actual_label = self.predict_actual[1]

    def calculate_metrics(self):
        """
           多分类任务计算评价指标时 参数 average 可选
                            1）micro：先计算整体FP，FN，TP，TN，再计算指标。
                            因多分类的FP+TP 或FN+TN 都等于样本总数，所以acc，prec，recall，f1相等
                            2）macro：先分别计算每一类的FP，FN，TP，TN，计算指标，再对所有类的相应指标取平均作为总体指标
        """
        if len(np.unique(self.actual_label)) == 2:
            self.predict_label[self.predict_label >= 0.5] = 1
            self.predict_label[self.predict_label < 0.5] = 0
        self.accuracy = metrics.accuracy_score(self.actual_label, self.predict_label)
        self.precision = metrics.precision_score(self.actual_label, self.predict_label, average=self.average)
        self.recall = metrics.recall_score(self.actual_label, self.predict_label, average=self.average)
        self.f1 = metrics.f1_score(self.actual_label, self.predict_label, average=self.average)

    def dump(self):
        self.logger.info('accuracy: %s' % self.accuracy)
        self.logger.info('precision: %s' % self.precision)
        self.logger.info('recall: %s' % self.recall)
        self.logger.info('f1: %s' % self.f1)
        with open(self.out_file, 'w', encoding='utf-8') as f:
            f.seek(0)
            f.write('=========================================== \n')
            f.write('accuracy: %s \n' % self.accuracy)
            f.write('precision: %s \n' % self.precision)
            f.write('recall: %s \n' % self.recall)
            f.write('f1: %s \n' % self.f1)
            f.truncate()
        f.close()

    def run(self):
        self.init()
        self.load_y()
        self.calculate_metrics()
        self.dump()
