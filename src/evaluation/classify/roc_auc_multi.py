# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from ..base import Base
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp


class RocAucMulti(Base):
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
        self.classes = self.dic_engine['classes']
        self.n_classes = len(self.classes)
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()

    def load_y(self):
        self.predict_actual = np.load(self.predicted_actual_path)
        self.predict_label = self.predict_actual[0]
        self.actual_label = self.predict_actual[1]
        # self.logger.info(self.predict_label)
        # self.logger.info(self.actual_label)

    def calculate_roc_auc(self):
        # Binarize the label list
        self.actual_label = label_binarize(self.actual_label, classes=self.classes)
        self.predict_label = label_binarize(self.predict_label, classes=self.classes)
        self.logger.info(self.predict_label.ravel())
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = self.n_classes
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.actual_label[:, i], self.predict_label[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.actual_label.ravel(), self.predict_label.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc

    def plot_and_dump(self):
        n_classes = self.n_classes
        fpr = self.fpr
        tpr = self.tpr
        roc_auc = self.roc_auc
        lw = 2
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(self.out_file, dpi=256)

    def run(self):
        self.init()
        self.load_y()
        self.calculate_roc_auc()
        self.plot_and_dump()
