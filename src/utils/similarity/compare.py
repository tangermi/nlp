# -*-coding:utf8-*-
from preprocess.preprocess import Preprocess
from feature.features import FeatureBuilder
from similarity.similarity import Similarity


# 判断两个文档是否重复
# INPUT: 文档1 + 文档2 + 停用词表 + 特征词典 + 模式选择 + 阈值
# OUTPUT: 输出两篇文档是否重复及相似度
class Compare:
    def __init__(self, doc_path_1, doc_path_2, stopword_path, word_dict, threshold):
        self.doc_path_1 = doc_path_1
        self.doc_path_2 = doc_path_2
        self.stopword_path = stopword_path
        self.word_dict = word_dict
        self.threshold = threshold
        self.preprocess = Preprocess

    # 对比相似度，有2种算法可以选择
    def compare(self, doc_fl_1, doc_fl_2, algo='cosine'):
        threshold = self.threshold
        if 'cosine' == algo:
            print('Matching by VSM + cosine distance')
            dist = Similarity.cosine_distance_nonzero(doc_fl_1.feat_vec, doc_fl_2.feat_vec, norm=False)
            if dist > float(threshold):
                print('Matching Result:\t<True:%s>' % dist)
            else:
                print('Matching Result:\t<False:%s>' % dist)
        elif 'simhash' == algo:
            print('Matching by Simhash + hamming distance')
            dist = Similarity.hamming_distance(doc_fl_1.fingerprint, doc_fl_2.fingerprint)
            if dist < float(threshold):
                print('Matching Result:\t<True:%s>' % dist)
            else:
                print('Matching Result:\t<False:%s>' % dist)

    def run(self, mode):
        doc_token_1, doc_token_2 = self.preprocess.preprocess(self.doc_path_1, self.doc_path_2, self.stopword_path)
        dict_file = 'dict.txt'
        self.preprocess.token2word_dict_tf(doc_token_1 + doc_token_2, dict_file)
        fb = FeatureBuilder(dict_file)
        doc_fl_1, doc_fl_2 = fb.dict2feature(doc_token_1, doc_token_2)
        self.compare(doc_fl_1, doc_fl_2, mode)
