# -*- coding:utf-8 -*-
from utils.segment.segment import Segmentor
import pickle
import numpy as np


# 模型数据
# 1.模型
# 2.feature_words
# 3.对应类别
class _MultinomialNB:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_path']
        self.feature_words_path = dic_config['feature_words_path']

    def load(self):
        self.nb = pickle.load(open(self.model_path, 'rb'))
        self.feature_words = np.load(self.feature_words_path)

    @staticmethod
    def text_features_(text, feature_words):  # 出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    # 特征化
    def feature(self, doc):
        data_list = []
        for text in doc:
            segmentor = Segmentor()
            word_list = segmentor.cut(text, '-j')  # 精简模式，返回一个可迭代的generator
            data_list.append(word_list)
        input_feature_list = [self.text_features_(text, self.feature_words) for text in data_list]

        self.input_feature_list = input_feature_list

    # 预测
    def _predict(self, text):
        self.feature([text])
        nums = self.nb.predict(self.input_feature_list)
        return nums
