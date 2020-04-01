# -*- coding: utf-8 -*-
import numpy as np
import csv
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from ..base import Base
import os


class Fs_each_class(Base):
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
        self.classes = self.dic_engine['classes']
        self.beta = self.dic_score['beta']
        self.average = self.dic_score['average']
        self.predicted_actual_path = self.dic_engine['_in']
        out_file = 'f_score.txt'
        if self.dic_score:
            out_file = self.dic_score.get('out_file', 'f_score.txt')
        self.out_file = os.path.join(self.dic_engine['_out'], out_file)

    def load_y(self):
        self.predict_actual = np.load(self.predicted_actual_path)
        self.predict_label = self.predict_actual[0]
        self.actual_label = self.predict_actual[1]

    def calculate_precision_recall_fscore_support(self):
        if len(np.unique(self.actual_label)) == 2:
            self.predict_label[self.predict_label >= 0.5] = 1
            self.predict_label[self.predict_label < 0.5] = 0
        self.precision, self.recall, self.fbeta, self.support = precision_recall_fscore_support(
            self.actual_label, self.predict_label, beta=self.beta, labels=list(self.classes), average=self.average
        )

    def dump(self):
        csvfile = open(self.out_file, 'wt', encoding="utf-8")
        csvfile.seek(0)
        writer = csv.writer(csvfile, delimiter=",")
        header = ['class', 'precision', 'recall', 'f-score', 'support']
        csvrow1 = np.array(self.classes).tolist()
        csvrow2 = self.precision.tolist()
        csvrow3 = self.recall.tolist()
        csvrow4 = self.fbeta.tolist()
        csvrow5 = self.support.tolist()
        writer.writerow(header)
        writer.writerows(zip(csvrow1, csvrow2, csvrow3, csvrow4, csvrow5))
        csvfile.truncate()
        csvfile.close()
        data = pd.read_csv(self.out_file)
        self.logger.info('\n %s' % data.head(len(self.classes)))

    def run(self):
        self.init()
        self.load_y()
        self.calculate_precision_recall_fscore_support()
        self.dump()

