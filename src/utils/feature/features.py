# -*-coding:utf8-*-
from similarity.simhash import SimhashBuilder
from similarity.similarity import Similarity


class FeatureBuilder:
    def __init__(self, word_dict_path):
        self.word_dict = {}
        self.word_dict_path = word_dict_path

    # 根据词典把文档的词转化为features
    def dict2feature(self, doc_token_1, doc_token_2):
        print('Loading word dict...')
        # Load word list from word_dict
        word_list = []
        with open(self.word_dict_path, 'r', encoding='utf-8') as ins:
            for line in ins.readlines():
                    word_list.append(line.split()[1])
                    # Build unicode string word dict
        for idx, ascword in enumerate(word_list):
            self.word_dict[ascword] = idx
            # Build nonzero-feature
        doc_feat_1 = self.compute(doc_token_1)
        doc_feat_2 = self.compute(doc_token_2)

        # Init simhash_builder
        smb = SimhashBuilder(word_list)

        doc_fl_1 = DocFeatLoader(smb, doc_feat_1)
        doc_fl_2 = DocFeatLoader(smb, doc_feat_2)

        return doc_fl_1, doc_fl_2

    # 把文字字典转换为（序号，词频）的列表 比如[(0, 37), (1, 14), (2, 17)]
    def compute(self, token_list):
        feature = [0] * len(self.word_dict)
        for token in token_list:
            feature[self.word_dict[token]] += 1
        feature_nonzero = [(idx, value) for idx, value in enumerate(feature) if value > 0]
        return feature_nonzero

    def _add_word(self, word):
        if not word in self.word_dict:
            self.word_dict[word] = len(self.word_dict)

    def update_words(self, word_list=[]):
        for word in word_list:
            self._add_word(word)


# 根据特征向量文件，生成norm_vector和fingerprint。 norm_vector用作cosine similarity计算， fingerprint用做hash similarity计算。
class DocFeatLoader:
    def __init__(self, simhash_builder, feat_nonzero):
        norm_vector_nonzero = Similarity.norm_vector_nonzero
        self.feat_vec = feat_nonzero
        self.feat_vec = norm_vector_nonzero(self.feat_vec)
        self.fingerprint = simhash_builder.sim_hash_nonzero(self.feat_vec)
