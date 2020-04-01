# -*- coding:utf-8 -*-
import numpy as np
from keras.utils import np_utils


def one_hot(seg_list):
    # 生成one-hot
    vocab = sorted(list(set(seg_list)))
    word_to_int = dict((w, i) for i, w in enumerate(vocab))
    int_to_word = dict((i, w) for i, w in enumerate(vocab))

    n_words = len(seg_list)  # 总词量
    n_vocab = len(vocab)  # 词表长度
    print('总词汇量：', n_words)
    print('词表长度：', n_vocab)

    seq_length = 100  # 句子长度
    dataX = []
    dataY = []
    for i in range(0, n_words - seq_length, 1):
        seq_in = seg_list[i:i + seq_length]
        seq_out = seg_list[i + seq_length]
        dataX.append([word_to_int[word] for word in seq_in])
        dataY.append(word_to_int[seq_out])

    n_simples = len(dataX)
    print('样本数量：', n_simples)
    X = np.reshape(dataX, (n_simples, seq_length))
    y = np_utils.to_categorical(dataY)
    return dataX, dataY, n_vocab, seq_length, int_to_word
