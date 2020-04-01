# -*- coding:utf-8 -*-
from snownlp import SnowNLP


# 输出每个部分的tf和idf的值,以及判断单词与文档的相似度
if __name__ == '__main__':
    text = [[u'这篇', u'文章'],
                 [u'那篇', u'论文'],
                 [u'这个']]

    s = SnowNLP(text)
    print(s.tf)
    print(s.idf)
    print(s.sim([u'文章']))  # [0.3756070762985226, 0, 0]