# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import csgraph_from_dense
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra


def csgraph():
    # 稠密，蒙板和稀疏表示
    G_dense = np.array([[0, 2, 1],
                        [2, 0, 0],
                        [1, 0, 0]])

    G_masked = np.ma.masked_values(G_dense, 0)
    G_sparse = csr_matrix(G_dense)
    print(G_sparse.data)

    # 使用蒙版或稀疏表示来消除歧义
    G2_data = np.array([
        [np.inf, 2, 0],
        [2, np.inf, np.inf],
        [0, np.inf, np.inf]
    ])
    G2_sparse = csgraph_from_dense(G2_data, null_value=np.inf)
    print(G2_sparse.data)

    # 获取单词列表
    wordlist = open('/apps/data/ai_nlp_testing/raw/scipy_exercises/words.txt').read().split()
    print(len(wordlist))

    # 现在想看长度为3的单词，选择正确长度的单词
    word_list = [word for word in wordlist if len(word) == 3]
    word_list = [word for word in word_list if word[0].islower()]
    word_list = [word for word in word_list if word.isalpha()]
    # word_list = map(str.lower, word_list)
    print(len(word_list))

    # 这些单词中的每一个都将成为图中的一个节点，我们将创建连接与每对单词关联的节点的边，这些节点之间的差异只有一个字母。
    word_list = np.asarray(word_list)
    print(word_list.dtype)
    word_list.sort()

    word_bytes = np.ndarray((word_list.size, word_list.itemsize),
                            dtype='int8',
                            buffer=word_list.data)
    print(word_bytes.shape)

    # 我们将使用每个点之间的汉明距离来确定连接哪些单词对
    hamming_dist = pdist(word_bytes, metric='hamming')
    graph = csr_matrix(squareform(hamming_dist < 1.5 / word_list.itemsize))

    # 使用最短路径搜索来查找图形中任何两个单词之间的路径
    i1 = word_list.searchsorted('ape')
    i2 = word_list.searchsorted('man')
    print(word_list[i1], word_list[i2])

    # 在图中找到这两个索引之间的最短路径
    distances, predecessors = dijkstra(graph, indices=i1, return_predecessors=True)
    print(distances[i2])

    # 使用算法返回的前辈来重构这条路径
    path = []
    i = i2

    while i != i1:
        path.append(word_list[i])
        i = predecessors[i]

    path.append(word_list[i1])
    print(path[::-1])


if __name__ == '__main__':
    csgraph()
