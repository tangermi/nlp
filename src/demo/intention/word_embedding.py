import os
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary


class Embedding(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


if __name__ == '__main__':
    # 训练word2vec模型
    sentences = Embedding('../data/')  # a memory-friendly iterator
