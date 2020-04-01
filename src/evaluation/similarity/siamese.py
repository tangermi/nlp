# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np
import tensorflow as tf
from utils.similarity.siamese import Similarity as s


class Siamese(Base):
    def __init__(self, dic_config={}, dic_engine={}, dic_score={}):
        self.dic_engine = dic_engine
        self.dic_score = dic_score
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)
        self.logger.info(self.dic_score)

    def init(self):
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test'])
        self.model_path = os.path.join(self.dic_engine['model_in'], self.dic_engine['model_file'])
        self.out_file = os.path.join(self.dic_engine['_out'], self.dic_score['out_file'])

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        with np.load(self.test_path) as test:
            self.te_pairs_1 = test['te_pairs_1']
            self.te_y_1 = test['te_y_1']
            self.te_pairs_2 = test['te_pairs_2']
            self.te_y_2 = test['te_y_2']
            self.te_pairs_3 = test['te_pairs_3']
            self.te_y_3 = test['te_y_3']

    def evaluate(self):
        model = self.model
        te_pairs_1 = self.te_pairs_1
        te_y_1 = self.te_y_1
        te_pairs_2 = self.te_pairs_2
        te_y_2 = self.te_y_2
        te_pairs_3 = self.te_pairs_3
        te_y_3 = self.te_y_3

        # compute final accuracy on training and test sets
        y_pred = model.predict([te_pairs_1[:, 0], te_pairs_1[:, 1]])
        self.te_acc_1 = s.compute_accuracy(te_y_1, y_pred)

        # predict test set 2
        y_pred = model.predict([te_pairs_2[:, 0], te_pairs_2[:, 1]])
        self.te_acc_2 = s.compute_accuracy(te_y_2, y_pred)

        # predict test set 3
        y_pred = model.predict([te_pairs_3[:, 0], te_pairs_3[:, 1]])
        self.te_acc_3 = s.compute_accuracy(te_y_3, y_pred)

    def dump(self):
        with open(self.out_file, 'w', encoding='utf-8') as f:
            f.seek(0)
            f.write('模型精确度：')
            f.write('\n* 测试集准确度: %0.2f%%' % (100 * self.te_acc_1))
            f.write('\n用["dress", "sneaker", "bag", "shirt"]这4个分类的物品测试(训练集中没出现过的品类):')
            f.write('\n* 测试准确度: %0.2f%%' % (100 * self.te_acc_2))
            f.write('\n用整个数据集来测试(包括训练集中没出现过的品类):')
            f.write('\n* 测试准确度: %0.2f%%' % (100 * self.te_acc_3))
            f.truncate()

    def run(self):
        self.init()
        self.load()
        self.evaluate()
        self.dump()
