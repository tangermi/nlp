# -*- coding:utf-8 -*-
from ..base import Base
import os
import pandas as pd
from utils.segment.segment import Segmentor
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle


'''
# 输入sogou数据集，内容为一条条的文本
# 输出为tfidf向量
'''
class _XGBoost(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_dir = self.dic_engine['_in']
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.vectorizer_path = os.path.join(self.dic_engine['out'], self.dic_engine['vectorizer_path'])
        self.tfidftransformer_path = os.path.join(self.dic_engine['out'], self.dic_engine['tfidftransformer_path'])
        self.train_feature_path = os.path.join(self.dic_engine['out'], self.dic_engine['train_feature'])
        self.test_feature_path = os.path.join(self.dic_engine['out'], self.dic_engine['test_feature'])
        self.stopwords_path = self.dic_engine['stop_word_file']

        self.vectorizer = CountVectorizer()
        self.tfidftransformer = TfidfTransformer()

    # 处理一个txt
    def load(self):
        data_dir = self.data_dir
        train_path = os.path.join(data_dir, 'train.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        train_df = pd.read_csv(train_path, sep=',', header=0, encoding='utf-8')
        test_df = pd.read_csv(test_path, sep=',', header=0, encoding='utf-8')
        # self.class_number = train_df['class'].unique().shape[0]
        train = [train_df['text'].values.tolist(), train_df['class'].values.tolist()]
        test = [test_df['text'].values.tolist(), test_df['class'].values.tolist()]

        self.train_content = self.segment_word(train[0])
        self.train_opinion = np.array(train[1])  # 需要numpy格式
        # self.logger.info(self.train_opinion.shape)

        self.test_content = self.segment_word(test[0])
        self.test_opinion = np.array(test[1])
        # self.logger.info(self.test_opinion.shape)

        self.logger.info('数据加载完成')

    def stop_words(self):
        stop_words_file = open(self.stopwords_path, 'r', encoding='utf-8')
        stopwords_list = []
        for line in stop_words_file.readlines():
            stopwords_list.append(line[:-1])
        return stopwords_list

    # 分词
    def segment_word(self, cont):
        stopwords_list = self.stop_words()
        res = []
        segment = Segmentor()
        for i in cont:
            text = ""
            word_list = segment.cut(i, '-j')
            for word in word_list:
                if word not in stopwords_list and word != '\r\n':
                    text += word
                    text += ' '
            res.append(text)
        return res

    # 转化为特征词数列
    def process(self):
        self.vectorizer = self.vectorizer.fit(self.train_content)
        tf = self.vectorizer.transform(self.train_content)
        self.tfidftransformer = self.tfidftransformer.fit(tf)
        tfidf = self.tfidftransformer.transform(tf)
        self.train_weight = tfidf.toarray()

        test_tf = self.vectorizer.transform(self.test_content)
        test_tfidf = self.tfidftransformer.transform(test_tf)
        self.test_weight = test_tfidf.toarray()

    def dump(self):
        # save train_weight
        np.savez(self.train_feature_path, train_feature=self.train_weight, train_class=self.train_opinion)
        # self.logger.info(self.train_weight.shape)
        # save test_weight
        np.savez(self.test_feature_path, test_feature=self.test_weight, test_class=self.test_opinion)
        # self.logger.info(self.test_weight.shape)

        # save feature list
        pickle.dump(self.vectorizer, open(self.vectorizer_path, "wb"))
        pickle.dump(self.tfidftransformer, open(self.tfidftransformer_path, "wb"))

    def run(self):
        self.init()
        self.load()
        self.process()
        self.dump()
