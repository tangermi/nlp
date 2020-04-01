# -*- coding: utf-8 -*-
"""
cofuse matrix
"""
from ..base import Base
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os


class ConfuseMatrix(Base):
    """params: actual_label: sample true label, 1d array or list[]
               predict_label: predicted data saved in directory predict，1d array or list[]
               classes: sample label_list

    """

    def __init__(self, dic_config={}, dic_engine={}, dic_score={}):
        self.dic_engine = dic_engine
        self.dic_score = dic_score
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)
        self.logger.info(self.dic_score)

    def init(self):
        self.predicted_actual_path = self.dic_engine['_in']
        self.classes = np.array(list(self.dic_engine['classes']))
        out_file = 'confuse_matrix.png'
        if self.dic_score:
            out_file = self.dic_score.get('out_file', 'confuse_matrix.png')
        self.out_file = os.path.join(self.dic_engine['_out'], out_file)

    def load_y(self):
        self.predict_actual = np.load(self.predicted_actual_path)
        self.predict_label = self.predict_actual[0]
        self.actual_label = self.predict_actual[1]

    def calculate_confuse_matrix(self):
        if len(np.unique(self.actual_label)) == 2:
            self.predict_label[self.predict_label >= 0.5] = 1
            self.predict_label[self.predict_label < 0.5] = 0
        self.cm_mat = confusion_matrix(self.actual_label, self.predict_label, labels=self.classes)
        self.cm_mat = self.cm_mat / (self.cm_mat.sum(axis=1) + 1e-8)[:, np.newaxis]
        self.cm_mat = self.cm_mat * 100
        self.cm_mat = self.cm_mat.astype('int')

    def plot_confuse_matrix(self):
        fig, ax = plt.subplots(figsize=(16, 12))
        im = ax.imshow(self.cm_mat, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(self.cm_mat.shape[1]),
               yticks=np.arange(self.cm_mat.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=self.classes, yticklabels=self.classes,
               title='confusion_matrix',
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        thresh = self.cm_mat.max() / 2.
        for i in range(self.cm_mat.shape[0]):
            for j in range(self.cm_mat.shape[1]):
                ax.text(j, i, format(self.cm_mat[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if self.cm_mat[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(self.out_file)

    def run(self):
        self.init()
        self.load_y()
        self.calculate_confuse_matrix()
        self.plot_confuse_matrix()
