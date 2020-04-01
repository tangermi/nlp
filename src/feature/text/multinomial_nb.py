# -*- coding:utf-8 -*-
from ..base import Base
import os
import pandas as pd
import numpy as np
from utils.segment.segment import Segmentor


'''
# 输入sogou数据集，内容为一条条的文本
# 输出特征词，以及特征词转会为向量的index

In: 地瓜真好吃
Out: [43, 344, 776, 65]
'''
class _MultinomialNB(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.train_feature_path = os.path.join(self.dic_engine['out'], self.dic_engine['train_feature'])
        self.test_feature_path = os.path.join(self.dic_engine['out'], self.dic_engine['test_feature'])

    # 处理一个txt
    def load(self):
        train_df = pd.read_csv(self.train_path, sep=',', header=0, encoding='utf-8')
        test_df = pd.read_csv(self.test_path, sep=',', header=0, encoding='utf-8')

        train = train_df['text'].values.tolist()

        self.train_data_list = []
        self.test_data_list = []
        for text in train:
            # 分词
            segment = Segmentor()
            word_cut =segment.cut(text, '-j')
            self.train_data_list.append(list(word_cut))
        test = test_df['text'].values.tolist()
        for text in test:
            word_cut = segment.cut(text, '-j')
            self.test_data_list.append(list(word_cut))
        self.train_class_list = train_df['class'].values.tolist()
        self.test_class_list = test_df['class'].values.tolist()
        self.stopword_set = set()

        all_words_dict = {}  # 统计训练集词频
        for word_list in self.train_data_list:
            for word in word_list:
                if word in all_words_dict.keys():
                    all_words_dict[word] += 1
                else:
                    all_words_dict[word] = 1

        # 根据键的值倒序排序
        all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
        all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
        self.all_words_list = list(all_words_list)  # 转换成列表

    def stopwords_set(self, words_file):
        with open(words_file, 'r', encoding='utf-8') as f:  # 打开文件
            for line in f.readlines():
                word = line.strip()  # 去回车
                if len(word) > 0:  # 有文本，则添加到words_set中
                    self.stopword_set.add(word)

    @staticmethod
    def text_features_(text, feature_words):  # 出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    def words_dict(self, all_words_list, deleteN, stopwords_set=set()):

        n = 1
        self.feature_words = []
        for t in range(deleteN, len(all_words_list), 1):
            if n > 2000:  # feature_words的维度为5000
                break
                # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词可以作为特征词
            if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                    all_words_list[t]) < 5:
                self.feature_words.append(all_words_list[t])
            n += 1
        feature_words_path = os.path.join(self.dic_engine['out'], self.dic_engine['feature_words'])
        np.save(feature_words_path, arr=self.feature_words)

    # 转化为特征词数列
    def process(self):
        # 生成停用词词典
        self.stopwords_set(self.dic_engine['stop_word_file'])
        # 生成一个特征词的词典
        self.words_dict(self.all_words_list, 0, self.stopword_set)
        self.train_feature_list = [self.text_features_(text, self.feature_words) for text in self.train_data_list]
        self.test_feature_list = [self.text_features_(text, self.feature_words) for text in self.test_data_list]

    def dump(self):
        np.savez(self.train_feature_path, train_feature_list=self.train_feature_list, train_class_list=self.train_class_list)
        np.savez(self.test_feature_path, test_feature_list=self.test_feature_list, test_class_list=self.test_class_list)

    def run(self):
        self.init()
        self.load()
        self.process()
        self.dump()
