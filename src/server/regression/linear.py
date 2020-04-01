# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from utils.feature.normalization import Normalization as norm


# 模型数据
# 1.模型
class Linear:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_path']
        self.mean_std_path = dic_config['mean_std_path']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path)
        with np.load(self.mean_std_path) as mean_std:
            self.mean = mean_std['mean']
            self.std = mean_std['std']

    def feature(self, dataset):
        dataset = np.array(dataset)
        # 归一化
        self.normed_train_data = norm.standard_score(dataset, self.mean, self.std)

    def _predict(self, test_data):
        self.feature(test_data)
        return self.model.predict(self.normed_train_data)
