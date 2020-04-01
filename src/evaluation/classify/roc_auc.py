# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from ..base import Base


class RocAuc(Base):
    """
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
        self.predicted_actual_path = self.dic_engine['_in']
        out_file = 'roc_auc.png'
        if self.dic_score:
            out_file = self.dic_score.get('out_file', 'roc_auc.png')
        self.out_file = os.path.join(self.dic_engine['_out'], out_file)
        # self.output_img = os.path.join(self.dic_engine['_out'], self.dic_score['image_path'])

    def load_y(self):
        self.predict_actual = np.load(self.predicted_actual_path)
        self.predict_label = self.predict_actual[0]
        self.actual_label = self.predict_actual[1]
        # self.logger.info(self.predict_label)
        # self.logger.info(self.actual_label)

    def calculate_roc_auc(self):
        # 计算fpr，tpr
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.actual_label.ravel(), self.predict_label.ravel())
        # 计算auc
        self.auc = metrics.auc(self.fpr, self.tpr)

    def plot_and_dump(self):
        # self.logger.info('auc: %s' % self.auc)
        # with open(self.out_file, 'a+', encoding='utf-8') as f:
        #     f.write('auc: %s \n' % self.auc)
        # f.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.fpr, self.tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % self.auc)
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.title(u'ROC-AUC', fontsize=17)
        plt.savefig(self.out_file, dpi=256)

    def run(self):
        self.init()
        self.load_y()
        self.calculate_roc_auc()
        self.plot_and_dump()
