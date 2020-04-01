# -*- coding:utf-8 -*-
from ..base import Base
import numpy as np
from keras.preprocessing import sequence
import os


'''
# 处理的数据集为imdb电影评论数据集，数据集已被word_to_index处理，tensorfolow提供数据与word_to_index的dictionary序列
# 输入数据为电影评论，以word_to_index后的数字序列表示[3, 756, 4334, 233, 67]
# 输出的features是用0作为补充padding后的ngram数组。[0, 0, 0, ... 21000, 1565, 95, 9]

In: [3, 756, 4334, 233, 67]   输入为单词的indices
Out: [0, 0, 0, ... 21000, 1565, 95, 9]   输出为新的ngram构成的indices
'''


class Fasttext(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.hyperparams = self.dic_engine['hyperparams']

        self.train_feature_path = os.path.join(self.dic_engine['out'], self.dic_engine['train_feature'])
        self.test_feature_path = os.path.join(self.dic_engine['out'], self.dic_engine['test_feature'])
        self.max_features_path = os.path.join(self.dic_engine['out'], self.dic_engine['max_features'])

    @staticmethod
    def create_ngram_set(input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    @staticmethod
    def add_ngram(sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.
        Example: adding bi-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
        Example: adding tri-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    def load(self):
        with np.load(self.train_path, allow_pickle=True) as train:
            self.x_train = train['x_train'].tolist()
            self.y_train = train['y_train'].tolist()

        with np.load(self.test_path, allow_pickle=True) as test:
            self.x_test = test['x_test'].tolist()
            self.y_test = test['y_test'].tolist()

    def process(self):
        ngram_range = self.hyperparams['ngram_range']
        maxlen = self.hyperparams['maxlen']
        max_features = self.hyperparams['max_features']

        if ngram_range > 1:
            self.logger.info('Adding {}-gram features'.format(ngram_range))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in self.x_train:
                for i in range(2, ngram_range + 1):
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)
            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = max_features + 1
            token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}

            # max_features is the highest integer that could be found in the dataset.
            self.max_features = np.max(list(indice_token.keys())) + 1

            # Augmenting x_train and x_test with n-grams features
            self.x_train = self.add_ngram(self.x_train, token_indice, ngram_range)
            self.x_test = self.add_ngram(self.x_test, token_indice, ngram_range)
            self.logger.info('Average train sequence length: {}'.format(np.mean(list(map(len, self.x_train)), dtype=int)))
            self.logger.info('Average test sequence length: {}'.format(np.mean(list(map(len, self.x_test)), dtype=int)))

        self.x_train = sequence.pad_sequences(self.x_train, maxlen=maxlen)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=maxlen)

    def dump(self):
        # save feature list
        np.savez(self.train_feature_path, x_train=self.x_train, y_train=self.y_train)
        np.savez(self.test_feature_path, x_test=self.x_test, y_test=self.y_test)
        np.save(self.max_features_path, arr=self.max_features)

    def run(self):
        self.init()
        self.load()
        self.process()
        self.dump()
