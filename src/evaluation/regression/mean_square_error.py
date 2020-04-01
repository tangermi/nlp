# -*- coding:utf-8 -*-
from ..base import Base
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class Linear(Base):
    def __init__(self, dic_config={}, dic_engine={}, dic_score={}):
        self.dic_engine = dic_engine
        self.dic_score = dic_score
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)
        self.logger.info(self.dic_score)

    def init(self):
        self.test_path = self.dic_engine['_in']
        self.model_path = os.path.join(self.dic_engine['model_in'], self.dic_engine['model_file'])
        self.test_labels = None
        self.normed_test_data = None
        self.outimg_path = os.path.join(self.dic_engine['out'], self.dic_score['out_img'])
        self.out_file = os.path.join(self.dic_engine['out'], self.dic_score['out_file'])
        self.model = None

    def load(self):
        with np.load(self.test_path) as test:
            self.normed_test_data = test['normed_test_data']
            self.test_labels = test['test_labels']
        self.model = tf.keras.models.load_model(self.model_path)

    def evaluate(self):
        model = self.model
        normed_test_data = self.normed_test_data
        test_labels = self.test_labels
        loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
        self.mae = mae
        self.logger.info("测试集平均绝对误差(MAE): {:5.2f} MPG".format(mae))
        # 测试集平均绝对误差(MAE):  1.90 MPG

        test_pred = model.predict(normed_test_data).flatten()

        # 解决中文乱码问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.scatter(test_labels, test_pred)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        plt.plot([-100, 100], [-100, 100])
        plt.savefig(self.outimg_path)

    def dump(self):
        with open(self.out_file, 'w', encoding='utf-8') as f:
            f.seek(0)
            f.write("测试集平均绝对误差(MAE): {:5.2f} MPG".format(self.mae))
            f.truncate()

    def run(self):
        self.init()
        self.load()
        self.evaluate()
        self.dump()
