# -*- coding:utf-8 -*-
import pickle
import xgboost as xgb
from utils.segment.segment import Segmentor


# 模型数据
# 1.模型
# 2.feature_words
# 3.对应类别
class _XGBoost:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_path']
        self.vectorizer_path = dic_config['vectorizer_path']
        self.tfidftransformer_path = dic_config['tfidftransformer_path']
        self.stopwords_path = dic_config['stopwords_path']

    def load(self):
        self.bst = pickle.load(open(self.model_path, 'rb'))
        self.vectorizer = pickle.load(open(self.vectorizer_path, "rb"))
        self.tfidftransformer = pickle.load(open(self.tfidftransformer_path, "rb"))

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

    def feature(self, doc):
        data_list = []
        for text in doc:
            data_list.append(text)
        self.doc_content = self.segment_word(data_list)
        tf = self.vectorizer.transform(self.doc_content)
        tfidf = self.tfidftransformer.transform(tf)
        self.doc_weight = tfidf.toarray()

    def _predict(self):
        ddoc = xgb.DMatrix(self.doc_weight)
        preds = self.bst.predict(ddoc)
        return preds
