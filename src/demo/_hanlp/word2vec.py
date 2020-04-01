# -*- coding:utf-8 -*-

from pyhanlp import *
import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH
from test_utility import *

class HanlpWord2Vec:
    def __init__(self):
        self.CoreSynonymDictionary = JClass("com.hankcs.hanlp.dictionary.CoreSynonymDictionary")
        self.WordVectorModel = JClass('com.hankcs.hanlp.mining.word2vec.WordVectorModel')
        self.DocVectorModel = JClass('com.hankcs.hanlp.mining.word2vec.DocVectorModel')
        self.model_path = '/apps/data/ai_nlp_testing/raw/corpus_for_word2doc/hanlp-wiki-vec-zh/hanlp-wiki-vec-zh.txt'
        self.word2vec = self.WordVectorModel(self.model_path)
        self.doc2vec = self.DocVectorModel(self.word2vec)

    # 语义距离
    def word_similarity(self, document1, document2):
        return self.CoreSynonymDictionary.similarity(document1, document2)

    def word_distance(self, document1, document2):
        return self.CoreSynonymDictionary.distance(document1, document2)

    def document_similarity(self, document1, document2):
        return self.doc2vec.similarity(document1, document2)

    def nearest_word(self, keyword):
        return self.word2vec.nearest(keyword)

    def nearest_document(self, keyword, document):
        for i, sentence in enumerate(document):
            self.doc2vec.addDocument(i, sentence)
        return self.doc2vec.nearest(keyword) # 通过 docs[res.getKey().intValue()] 和 res.getValue().floatValue() 取到每个文档结果 以及 相关的值



# test
if __name__ == '__main__':

    keywords = ['体育', '农业', '我要看比赛', '要不做饭吧', '山东', '江苏', '上班']
    docs = ["山东苹果丰收", "农民在江苏种水稻", "奥运会女排夺冠", "java程序员", "中国足球失败", "农家小炒肉"]

    word2vec = HanlpWord2Vec()
    print(f"“{keywords[4]}”与“{keywords[5]}”之间的距离与相似度:")
    word2vec.word_distance(keywords[4],keywords[5])
    word2vec.word_similarity(keywords[5],keywords[6])
    print(f"“{keywords[4]}”与“{keywords[6]}”之间的距离与相似度:")
    print(word2vec.word_distance(keywords[4], keywords[6]))
    print(word2vec.word_similarity(keywords[4], keywords[6]))
    print('\n找出相似的词语','-'*20)
    for keyword in keywords:
        print(f'{keyword}的相似词语')
        print(word2vec.nearest_word(keyword))
    print('\n找出相关文档','-'*20)
    for keyword in keywords:
        print(f"\n“{keyword}”的相关文档：")
        for res in word2vec.nearest_document(keyword,docs):
            print('%s = %.2f' % (docs[res.getKey().intValue()], res.getValue().floatValue()))
    print('\n文档相似度计算','-' * 20)
    docs4sim = ['山西副省长贪污腐败开庭', '陕西村干部受贿违纪', '股票基金增长']
    print(f"《{docs4sim[0]}》和《{docs4sim[1]}》的相似度:")
    print(word2vec.document_similarity(docs4sim[0], docs4sim[1]))
    print(f"《{docs4sim[0]}》和《{docs4sim[2]}》的相似度:")
    print(word2vec.document_similarity(docs4sim[0], docs4sim[2]))
